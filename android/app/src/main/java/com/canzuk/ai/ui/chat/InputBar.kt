package com.canzuk.ai.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
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
    enabled: Boolean = true,
) {
    var text by remember { mutableStateOf("") }

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

        IconButton(
            onClick = {
                if (text.isNotBlank()) {
                    onSend(text)
                    text = ""
                }
            },
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(if (text.isNotBlank()) CanzukRed else TextDim.copy(alpha = 0.2f)),
            enabled = text.isNotBlank() && enabled,
        ) {
            Icon(
                Icons.Filled.ArrowUpward,
                contentDescription = "Send",
                tint = CanzukWhite,
            )
        }
    }
}
