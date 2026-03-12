package com.canzuk.ai.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Mic
import androidx.compose.material.icons.outlined.Folder
import androidx.compose.material.icons.outlined.Groups
import androidx.compose.material.icons.outlined.Psychology
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import com.canzuk.ai.ui.theme.*

data class MoreItem(val label: String, val subtitle: String, val icon: ImageVector)

val moreItems = listOf(
    MoreItem("Voice", "Hands-free conversation", Icons.Outlined.Mic),
    MoreItem("Files", "Upload and analyse documents", Icons.Outlined.Folder),
    MoreItem("Cowork", "Multi-turn task execution", Icons.Outlined.Groups),
    MoreItem("Introspection", "Live model stats", Icons.Outlined.Psychology),
)

@Composable
fun MoreScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp),
    ) {
        Text(
            "More",
            style = MaterialTheme.typography.headlineMedium,
            color = CanzukRed,
            modifier = Modifier.padding(bottom = 16.dp),
        )

        moreItems.forEach { item ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                ),
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        item.icon,
                        contentDescription = item.label,
                        tint = CanzukBlue,
                        modifier = Modifier.size(28.dp),
                    )
                    Spacer(Modifier.width(16.dp))
                    Column {
                        Text(item.label, style = MaterialTheme.typography.titleMedium, color = TextPrimary)
                        Text(item.subtitle, style = MaterialTheme.typography.labelSmall, color = TextDim)
                    }
                }
            }
        }
    }
}
