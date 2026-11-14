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

import com.google.ai.edge.gallery.data.Model
import java.io.OutputStream
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.util.UUID
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json

private const val TAG = "LocalApiRouter"

/**
 * Runtime boundary for the HTTP API. The implementation is deliberately supplied by the app's
 * model/session layer so the HTTP transport does not depend on a UI ViewModel or a particular
 * LiteRT-LM runtime generation.
 */
internal interface LocalApiInference {
  fun listModels(): List<Model>
  suspend fun complete(request: ChatCompletionRequest): String
  suspend fun stream(request: ChatCompletionRequest, onToken: (String) -> Unit)
}

internal class LocalApiRouter(
  private val inference: LocalApiInference,
  private val apiKey: String? = null,
) {
  private val json = Json {
    ignoreUnknownKeys = true
    isLenient = false
    encodeDefaults = true
  }

  fun handle(request: HttpRequest, socket: Socket) {
    Log.i(TAG, "Request ${request.method} ${request.path} from ${request.remoteAddress}")
    if (request.method == "OPTIONS") {
      writeResponse(socket, 204, "", "text/plain")
      return
    }
    if (!authorized(request)) {
      writeJson(socket, 401, ApiError("Unauthorized", "authentication_error"))
      return
    }

    when {
      request.method == "GET" && request.path == "/health" ->
        writeJson(socket, 200, HealthResponse("ok"))

      request.method == "GET" && request.path == "/v1/models" ->
        writeJson(socket, 200, ModelsResponse(inference.listModels().map { ApiModel(it.name) }))

      request.method == "POST" && request.path == "/v1/chat/completions" ->
        handleChatCompletion(request, socket)

      else -> writeJson(socket, 404, ApiError("Not found", "not_found"))
    }
  }

  private fun authorized(request: HttpRequest): Boolean {
    if (apiKey.isNullOrBlank()) return true
    return request.headers["authorization"] == "Bearer $apiKey"
  }

  private fun handleChatCompletion(request: HttpRequest, socket: Socket) {
    val chatRequest = try {
      json.decodeFromString<ChatCompletionRequest>(request.body)
    } catch (error: Exception) {
      writeJson(socket, 400, ApiError("Invalid chat completion request: ${error.message}", "bad_request"))
      return
    }

    if (chatRequest.messages.isEmpty()) {
      writeJson(socket, 400, ApiError("messages must not be empty", "bad_request"))
      return
    }
    if (chatRequest.model.isBlank()) {
      writeJson(socket, 400, ApiError("model is required", "bad_request"))
      return
    }

    try {
      if (chatRequest.stream) {
        streamChatCompletion(socket, chatRequest)
      } else {
        val output = runBlocking { inference.complete(chatRequest) }
        writeJson(
          socket,
          200,
          ChatCompletionResponse(
            id = "chatcmpl-${UUID.randomUUID()}",
            model = chatRequest.model,
            choices = listOf(ChatChoice(ChatMessage("assistant", output))),
          ),
        )
      }
    } catch (error: IllegalArgumentException) {
      writeJson(socket, 400, ApiError(error.message ?: "Invalid request", "invalid_request_error"))
    } catch (error: Exception) {
      writeJson(socket, 500, ApiError(error.message ?: "Inference failed", "server_error"))
    }
  }

  private fun streamChatCompletion(socket: Socket, request: ChatCompletionRequest) {
    val output = socket.getOutputStream()
    writeHeaders(output, 200, "text/event-stream", chunked = false, contentLength = null)
    val responseId = "chatcmpl-${UUID.randomUUID()}"
    try {
      runBlocking {
        inference.stream(request) { token ->
          val chunk = ChatCompletionChunk(
            id = responseId,
            model = request.model,
            choices = listOf(ChatStreamChoice(delta = ChatDelta(content = token))),
          )
          writeSse(output, json.encodeToString(chunk))
        }
      }
      val finalChunk = ChatCompletionChunk(
        id = responseId,
        model = request.model,
        choices = listOf(ChatStreamChoice(delta = ChatDelta(), finish_reason = "stop")),
      )
      writeSse(output, json.encodeToString(finalChunk))
      writeSse(output, "[DONE]")
    } catch (error: Exception) {
      // Headers have already been sent. Signal the failure as an SSE data event, then close.
      val errorJson = json.encodeToString(ApiError(error.message ?: "Inference failed", "server_error"))
      writeSse(output, errorJson)
    }
  }

  private fun writeSse(output: OutputStream, data: String) {
    output.write("data: $data\n\n".toByteArray(StandardCharsets.UTF_8))
    output.flush()
  }

  private inline fun <reified T> writeJson(socket: Socket, status: Int, value: T) {
    writeResponse(socket, status, json.encodeToString(value), "application/json")
  }

  private fun writeResponse(socket: Socket, status: Int, body: String, contentType: String) {
    val output = socket.getOutputStream()
    val bytes = body.toByteArray(StandardCharsets.UTF_8)
    writeHeaders(output, status, contentType, chunked = false, contentLength = bytes.size)
    output.write(bytes)
    output.flush()
  }

  private fun writeHeaders(
    output: OutputStream,
    status: Int,
    contentType: String,
    chunked: Boolean,
    contentLength: Int? = 0,
  ) {
    val reason = when (status) {
      200 -> "OK"
      204 -> "No Content"
      400 -> "Bad Request"
      401 -> "Unauthorized"
      404 -> "Not Found"
      else -> "Internal Server Error"
    }
    val headers = buildString {
      append("HTTP/1.1 $status $reason\r\n")
      append("Content-Type: $contentType; charset=utf-8\r\n")
      if (chunked) append("Transfer-Encoding: chunked\r\n")
      else if (contentLength != null) append("Content-Length: $contentLength\r\n")
      append("Connection: close\r\n")
      append("Access-Control-Allow-Origin: *\r\n")
      append("Access-Control-Allow-Headers: Authorization, Content-Type\r\n")
      append("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n")
      append("\r\n")
    }
    output.write(headers.toByteArray(StandardCharsets.UTF_8))
    output.flush()
  }
}

@Serializable
internal data class ChatCompletionRequest(
  val model: String,
  val messages: List<ChatMessage>,
  val stream: Boolean = false,
  val temperature: Double? = null,
  val max_tokens: Int? = null,
  val top_p: Double? = null,
  val top_k: Int? = null,
)

@Serializable
internal data class ChatMessage(val role: String, val content: String)

@Serializable
private data class ModelsResponse(val `object`: String = "list", val data: List<ApiModel>)

@Serializable
private data class ApiModel(
  val id: String,
  val `object`: String = "model",
  val owned_by: String = "local",
)

@Serializable
private data class HealthResponse(val status: String)

@Serializable
private data class ApiError(val error: String, val type: String)

@Serializable
private data class ChatCompletionResponse(
  val id: String,
  val `object`: String = "chat.completion",
  val created: Long = System.currentTimeMillis() / 1000,
  val model: String,
  val choices: List<ChatChoice>,
)

@Serializable
private data class ChatChoice(
  val message: ChatMessage,
  val index: Int = 0,
  val finish_reason: String = "stop",
)

@Serializable
private data class ChatCompletionChunk(
  val id: String,
  val `object`: String = "chat.completion.chunk",
  val created: Long = System.currentTimeMillis() / 1000,
  val model: String,
  val choices: List<ChatStreamChoice>,
)

@Serializable
private data class ChatStreamChoice(
  val delta: ChatDelta,
  val index: Int = 0,
  val finish_reason: String? = null,
)

@Serializable
private data class ChatDelta(val content: String? = null)
