package com.canzuk.ai.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import com.canzuk.ai.ui.theme.*

@Composable
fun InputBar(
    onSend: (String) -> Unit,
    onMicTap: () -> Unit = {},
    enabled: Boolean = true,
) {
    var text by remember { mutableStateOf("") }
    val hasText = text.isNotBlank()

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.Bottom,
    ) {
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            modifier = Modifier.weight(1f),
            placeholder = {
                Text("Ask CANZUK-AI...", color = TextDim)
            },
            shape = RoundedCornerShape(24.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = CanzukBlue,
                unfocusedBorderColor = TextDim.copy(alpha = 0.3f),
                cursorColor = CanzukRed,
                focusedTextColor = TextPrimary,
                unfocusedTextColor = TextPrimary,
            ),
            maxLines = 5,
            enabled = enabled,
        )

        Spacer(Modifier.width(8.dp))

        if (hasText) {
            // Send button — shown when text is present
            IconButton(
                onClick = {
                    onSend(text)
                    text = ""
                },
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(CanzukRed),
                enabled = enabled,
            ) {
                Icon(
                    Icons.Filled.ArrowUpward,
                    contentDescription = "Send",
                    tint = CanzukWhite,
                )
            }
        } else {
            // Mic button — shown when text field is empty
            IconButton(
                onClick = onMicTap,
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(CanzukRed),
                enabled = enabled,
            ) {
                Icon(
                    Icons.Filled.Mic,
                    contentDescription = "Voice mode",
                    tint = CanzukWhite,
                )
            }
        }
    }
}
