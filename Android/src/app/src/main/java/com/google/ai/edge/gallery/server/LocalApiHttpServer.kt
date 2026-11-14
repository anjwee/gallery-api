/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.server

import android.util.Log
import java.io.BufferedInputStream
import java.io.IOException
import java.io.OutputStream
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "LocalApiHttpServer"
private const val MAX_HEADER_BYTES = 64 * 1024
private const val MAX_BODY_BYTES = 8 * 1024 * 1024
private const val REQUEST_READ_TIMEOUT_MS = 30_000

/**
 * Minimal HTTP/1.1 transport for the on-device API.
 *
 * Routing and model inference deliberately live outside this class. The handler receives the
 * parsed request and the connected socket so it can write either a normal JSON response or an
 * SSE stream without buffering the complete response in memory.
 */
internal class LocalApiHttpServer(
  private val host: String,
  private val port: Int,
  private val maxConnections: Int = 4,
  private val handler: (HttpRequest, Socket) -> Unit,
) {
  private val running = AtomicBoolean(false)

  @Volatile private var serverSocket: ServerSocket? = null
  @Volatile private var acceptThread: Thread? = null
  @Volatile private var connectionExecutor: ExecutorService? = null

  @Synchronized
  fun start() {
    check(!running.get()) { "Local API server is already running" }
    require(port in 1..65535) { "Port must be between 1 and 65535" }

    val address = InetAddress.getByName(host)
    val socket = ServerSocket()
    socket.reuseAddress = true
    try {
      socket.bind(InetSocketAddress(address, port))
    } catch (error: Exception) {
      runCatching { socket.close() }
      throw error
    }

    serverSocket = socket
    connectionExecutor = Executors.newFixedThreadPool(maxConnections.coerceAtLeast(1))
    running.set(true)
    acceptThread = Thread({ acceptConnections(socket) }, "local-api-accept").apply {
      isDaemon = true
      start()
    }
    Log.i(TAG, "HTTP server listening on $host:$port")
  }

  @Synchronized
  fun stop() {
    running.set(false)
    runCatching { serverSocket?.close() }
    serverSocket = null
    connectionExecutor?.shutdownNow()
    connectionExecutor = null
    acceptThread = null
    Log.i(TAG, "HTTP server stopped")
  }

  fun isRunning(): Boolean = running.get()

  private fun acceptConnections(listener: ServerSocket) {
    while (running.get()) {
      try {
        val client = listener.accept()
        try {
          connectionExecutor?.execute { handleConnection(client) }
            ?: runCatching { client.close() }
        } catch (error: Exception) {
          runCatching { client.close() }
          if (running.get()) Log.e(TAG, "Unable to dispatch client connection", error)
        }
      } catch (error: SocketException) {
        if (running.get()) Log.e(TAG, "Accept failed", error)
      } catch (error: Exception) {
        if (running.get()) Log.e(TAG, "Unexpected accept error", error)
      }
    }
  }

  private fun handleConnection(socket: Socket) {
    socket.use { client ->
      try {
        client.soTimeout = REQUEST_READ_TIMEOUT_MS
        val request = readRequest(client)
        // Do not apply the request-read timeout to inference or a long-lived SSE response.
        client.soTimeout = 0
        handler(request, client)
      } catch (error: HttpRequestException) {
        runCatching {
          writeError(client.getOutputStream(), error.statusCode, error.message ?: "Bad request")
        }
      } catch (error: SocketException) {
        Log.d(TAG, "Client disconnected: ${error.message}")
      } catch (error: IOException) {
        Log.w(TAG, "I/O error while handling client", error)
      } catch (error: Exception) {
        Log.e(TAG, "Request handler failed", error)
        runCatching {
          writeError(client.getOutputStream(), 500, "Internal server error")
        }
      }
    }
  }

  private fun readRequest(socket: Socket): HttpRequest {
    val input = BufferedInputStream(socket.getInputStream())
    var headerBytesRead = 0

    fun readLine(): String {
      val bytes = ArrayList<Byte>()
      while (true) {
        val next = input.read()
        if (next == -1) throw HttpRequestException(400, "Unexpected end of request headers")
        headerBytesRead++
        if (headerBytesRead > MAX_HEADER_BYTES) {
          throw HttpRequestException(431, "Request headers are too large")
        }
        if (next == '\n'.code) break
        if (next != '\r'.code) bytes.add(next.toByte())
      }
      return bytes.toByteArray().toString(StandardCharsets.UTF_8)
    }

    val requestLine = readLine()
    val requestParts = requestLine.split(' ', limit = 3)
    if (requestParts.size != 3 || !requestParts[2].startsWith("HTTP/1.")) {
      throw HttpRequestException(400, "Invalid HTTP request line")
    }

    val headers = linkedMapOf<String, String>()
    while (true) {
      val line = readLine()
      if (line.isEmpty()) break
      val separator = line.indexOf(':')
      if (separator <= 0) throw HttpRequestException(400, "Invalid HTTP header")
      val name = line.substring(0, separator).trim().lowercase(Locale.US)
      val value = line.substring(separator + 1).trim()
      if (name.isEmpty()) throw HttpRequestException(400, "Invalid HTTP header name")
      headers[name] = value
    }

    if (headers.containsKey("transfer-encoding")) {
      throw HttpRequestException(501, "Transfer-Encoding is not supported")
    }
    val contentLength = headers["content-length"]?.toLongOrNull() ?: 0L
    if (contentLength < 0 || contentLength > MAX_BODY_BYTES) {
      throw HttpRequestException(413, "Request body is too large")
    }
    if (headers["content-length"] != null && headers["content-length"]?.toLongOrNull() == null) {
      throw HttpRequestException(400, "Invalid Content-Length")
    }

    val body = ByteArray(contentLength.toInt())
    var offset = 0
    while (offset < body.size) {
      val count = input.read(body, offset, body.size - offset)
      if (count < 0) throw HttpRequestException(400, "Incomplete request body")
      offset += count
    }

    return HttpRequest(
      method = requestParts[0].uppercase(Locale.US),
      path = requestParts[1].substringBefore('?'),
      version = requestParts[2],
      headers = headers,
      body = body.toString(StandardCharsets.UTF_8),
      remoteAddress = socket.inetAddress.hostAddress.orEmpty(),
    )
  }

  private fun writeError(output: OutputStream, status: Int, message: String) {
    val body = """{"error":"${message.replace("\\", "\\\\").replace("\"", "\\\"")}"}"""
      .toByteArray(StandardCharsets.UTF_8)
    val reason = when (status) {
      400 -> "Bad Request"
      413 -> "Payload Too Large"
      431 -> "Request Header Fields Too Large"
      501 -> "Not Implemented"
      else -> "Internal Server Error"
    }
    val headers = "HTTP/1.1 $status $reason\r\n" +
      "Content-Type: application/json; charset=utf-8\r\n" +
      "Content-Length: ${body.size}\r\n" +
      "Connection: close\r\n\r\n"
    output.write(headers.toByteArray(StandardCharsets.UTF_8))
    output.write(body)
    output.flush()
  }
}

internal data class HttpRequest(
  val method: String,
  val path: String,
  val version: String,
  val headers: Map<String, String>,
  val body: String,
  val remoteAddress: String,
)

private class HttpRequestException(
  val statusCode: Int,
  message: String,
) : IOException(message)
