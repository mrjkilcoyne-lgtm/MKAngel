package com.canzuk.ai.ui.chat

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.canzuk.ai.data.Message
import com.canzuk.ai.data.MessageRole
import com.canzuk.ai.ui.theme.*

@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val state by viewModel.state.collectAsState()
    val listState = rememberLazyListState()

    // Auto-scroll to bottom on new messages
    LaunchedEffect(state.messages.size, state.streamingContent) {
        if (state.messages.isNotEmpty()) {
            listState.animateScrollToItem(state.messages.size)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // Header — CANZUK in red, -AI in blue
        Surface(
            color = MaterialTheme.colorScheme.background,
            shadowElevation = 1.dp,
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .statusBarsPadding()
                    .padding(horizontal = 20.dp, vertical = 12.dp),
            ) {
                Text(
                    "CANZUK",
                    style = MaterialTheme.typography.headlineMedium,
                    color = CanzukRed,
                )
                Text(
                    "-AI",
                    style = MaterialTheme.typography.headlineMedium,
                    color = CanzukBlue,
                )
            }
        }

        // Messages
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(vertical = 8.dp),
        ) {
            items(state.messages) { message ->
                MessageBubble(message)
            }

            // Streaming indicator
            if (state.isGenerating && state.streamingContent.isNotEmpty()) {
                item {
                    MessageBubble(
                        Message(
                            role = MessageRole.ASSISTANT,
                            content = state.streamingContent,
                            isStreaming = true,
                        )
                    )
                }
            } else if (state.isGenerating) {
                item {
                    Row(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                        Text("Deriving...", color = TextDim, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }

        // Input
        InputBar(
            onSend = viewModel::send,
            enabled = !state.isGenerating,
        )
    }
}
