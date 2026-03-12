package com.canzuk.ai.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.canzuk.ai.ui.theme.*

@Composable
fun DocumentScreen() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Document", style = MaterialTheme.typography.headlineMedium, color = CanzukBlue)
            Spacer(Modifier.height(8.dp))
            Text("Long-form generation", style = MaterialTheme.typography.bodyMedium, color = TextDim)
            Spacer(Modifier.height(4.dp))
            Text("Essays, reports, analysis — guided by derivation trees", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
        }
    }
}
