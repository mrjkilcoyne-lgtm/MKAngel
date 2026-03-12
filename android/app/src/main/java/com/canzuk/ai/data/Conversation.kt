package com.canzuk.ai.data

import java.util.UUID

data class Conversation(
    val id: String = UUID.randomUUID().toString(),
    val title: String = "New conversation",
    val messages: List<Message> = emptyList(),
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
)
