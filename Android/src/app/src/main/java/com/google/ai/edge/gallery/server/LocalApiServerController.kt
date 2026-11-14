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

/**
 * Owns the HTTP listener lifecycle independently of Android UI components.
 *
 * The caller supplies the inference adapter and persists user preferences. The listener binds to
 * all network interfaces by default so devices on the same LAN can connect.
 */
internal class LocalApiServerController(
  private val inference: LocalApiInference,
) {
  private var server: LocalApiHttpServer? = null

  @Synchronized
  fun start(port: Int, apiKey: String? = null) {
    check(server == null) { "Local API server is already running" }
    require(port in 1024..65535) { "Port must be between 1024 and 65535" }

    val router = LocalApiRouter(inference = inference, apiKey = apiKey)
    val candidate =
      LocalApiHttpServer(
        host = "0.0.0.0",
        port = port,
        handler = router::handle,
      )
    candidate.start()
    server = candidate
  }

  @Synchronized
  fun stop() {
    server?.stop()
    server = null
  }

  @Synchronized
  fun isRunning(): Boolean = server?.isRunning() == true
}
