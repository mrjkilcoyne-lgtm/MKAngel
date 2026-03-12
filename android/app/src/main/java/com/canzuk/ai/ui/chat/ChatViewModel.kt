package com.canzuk.ai.ui.chat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.canzuk.ai.data.Message
import com.canzuk.ai.data.MessageRole
import com.canzuk.ai.engine.EngineViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

data class ChatUiState(
    val messages: List<Message> = emptyList(),
    val isGenerating: Boolean = false,
    val streamingContent: String = "",
)

class ChatViewModel(
    private val engine: EngineViewModel,
) : ViewModel() {

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    /**
     * SharedFlow that emits complete assistant responses produced by voice input.
     * The VoiceModeScreen collects this to feed responses into TTS playback.
     */
    private val _voiceResponse = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val voiceResponse: SharedFlow<String> = _voiceResponse.asSharedFlow()

    fun send(text: String) {
        if (text.isBlank()) return

        // Add user message
        val userMsg = Message(role = MessageRole.USER, content = text)
        _state.update { it.copy(messages = it.messages + userMsg, isGenerating = true, streamingContent = "") }

        // Stream response
        viewModelScope.launch {
            try {
                val buffer = StringBuilder()
                engine.stream(text).collect { token ->
                    buffer.append(token)
                    _state.update { it.copy(streamingContent = buffer.toString()) }
                }

                // Finalise assistant message
                val assistantMsg = Message(role = MessageRole.ASSISTANT, content = buffer.toString())
                _state.update {
                    it.copy(
                        messages = it.messages + assistantMsg,
                        isGenerating = false,
                        streamingContent = "",
                    )
                }
            } catch (e: Exception) {
                val errorMsg = Message(
                    role = MessageRole.ASSISTANT,
                    content = "Engine error: ${e.message ?: "unknown failure"}",
                )
                _state.update {
                    it.copy(
                        messages = it.messages + errorMsg,
                        isGenerating = false,
                        streamingContent = "",
                    )
                }
            }
        }
    }

    /**
     * Accept transcribed voice input and process it through the same GLM
     * engine pipeline used by text chat.
     *
     * The user message is appended to the conversation history so it
     * appears in the chat screen as well. Once the full response has been
     * assembled it is emitted on [voiceResponse] for TTS playback.
     */
    fun processVoiceInput(text: String) {
        if (text.isBlank()) return

        // Add user message from voice
        val userMsg = Message(role = MessageRole.USER, content = text)
        _state.update { it.copy(messages = it.messages + userMsg, isGenerating = true, streamingContent = "") }

        // Process through engine same as text input
        viewModelScope.launch {
            try {
                val buffer = StringBuilder()
                engine.stream(text).collect { token ->
                    buffer.append(token)
                    _state.update { it.copy(streamingContent = buffer.toString()) }
                }

                // Finalise assistant message
                val response = buffer.toString()
                val assistantMsg = Message(role = MessageRole.ASSISTANT, content = response)
                _state.update {
                    it.copy(
                        messages = it.messages + assistantMsg,
                        isGenerating = false,
                        streamingContent = "",
                    )
                }

                // Return response for TTS
                _voiceResponse.tryEmit(response)
            } catch (e: Exception) {
                val errorMsg = Message(
                    role = MessageRole.ASSISTANT,
                    content = "Engine error: ${e.message ?: "unknown failure"}",
                )
                _state.update {
                    it.copy(
                        messages = it.messages + errorMsg,
                        isGenerating = false,
                        streamingContent = "",
                    )
                }
            }
        }
    }
}
