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

    fun send(text: String) {
        if (text.isBlank()) return

        // Add user message
        val userMsg = Message(role = MessageRole.USER, content = text)
        _state.update { it.copy(messages = it.messages + userMsg, isGenerating = true, streamingContent = "") }

        // Stream response
        viewModelScope.launch {
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
        }
    }
}
