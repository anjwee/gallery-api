package com.google.ai.edge.gallery.server

import android.content.Context
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.runtime.runtimeHelper
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

private const val EPHEMERAL_INFERENCE = "aicore_ephemeral_inference"

internal interface LocalApiModelProvider {
  fun availableModels(): List<Model>
  fun selectedModel(): Model?
}

/** Keeps API inference state separate from the Model instance used by the chat UI. */
internal class LocalApiInferenceAdapter(
  context: Context,
  private val modelProvider: LocalApiModelProvider,
) : LocalApiInference {
  private val appContext = context.applicationContext

  override fun listModels(): List<Model> = modelProvider.availableModels()

  override suspend fun complete(request: ChatCompletionRequest): String {
    val result = StringBuilder()
    run(request) { result.append(it) }
    return result.toString()
  }

  override suspend fun stream(request: ChatCompletionRequest, onToken: (String) -> Unit) {
    run(request, onToken)
  }

  private suspend fun run(request: ChatCompletionRequest, onToken: (String) -> Unit = {}) {
    val source = modelProvider.availableModels().firstOrNull { it.name == request.model }
      ?: modelProvider.selectedModel()?.takeIf { it.name == request.model }
      ?: throw IllegalArgumentException("Model not available: " + request.model)
    require(source.isLlm) { "Selected model is not an LLM." }

    // Preserve the original storage directory while giving this runtime instance its own identity.
    val apiModel = source.copy(name = source.name + "_api_" + UUID.randomUUID())
    apiModel.normalizedName = source.normalizedName
    apiModel.configValues = source.configValues.toMap()
    apiModel.prevConfigValues = source.prevConfigValues.toMap()
    val helper = apiModel.runtimeHelper

    try {
      val loadError = withContext(Dispatchers.IO) {
        helper.loadModel(appContext, apiModel, "local_api", false, false)
      }
      if (loadError != null) throw IllegalStateException(loadError)

      val prompt = request.messages.joinToString("\n\n") { message ->
        val role = message.role.lowercase()
        val label = when (role) {
          "system" -> "System"
          "assistant" -> "Assistant"
          "user" -> "User"
          else -> message.role
        }
        label + ": " + message.content
      } + "\n\nAssistant:"

      val extraContext = buildMap {
        put(EPHEMERAL_INFERENCE, "true")
        request.max_tokens?.takeIf { it > 0 }?.let { put("max_output_tokens", it.toString()) }
      }

      suspendCancellableCoroutine { continuation ->
        val scope = CoroutineScope(currentCoroutineContext())
        continuation.invokeOnCancellation { helper.stopResponse(apiModel) }
        helper.runInference(
          model = apiModel,
          input = prompt,
          resultListener = { partial, done, _ ->
            if (partial.isNotEmpty()) onToken(partial)
            if (done && continuation.isActive) continuation.resume(Unit)
          },
          cleanUpListener = {},
          onError = { message ->
            if (continuation.isActive) {
              continuation.resumeWithException(IllegalStateException(message))
            }
          },
          coroutineScope = scope,
          extraContext = extraContext,
        )
      }
    } catch (error: CancellationException) {
      throw error
    } finally {
      withContext(Dispatchers.IO) { helper.cleanUp(apiModel) {} }
    }
  }
}
