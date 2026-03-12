package com.canzuk.ai.data

import java.util.UUID

enum class MessageRole { USER, ASSISTANT, SYSTEM }

data class Message(
    val id: String = UUID.randomUUID().toString(),
    val role: MessageRole,
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
    val isStreaming: Boolean = false,
    val domains: List<String> = emptyList(),
)
