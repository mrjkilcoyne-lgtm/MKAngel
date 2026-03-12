package com.canzuk.ai.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.canzuk.ai.ui.theme.*

@Composable
fun CanvasScreen() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Canvas", style = MaterialTheme.typography.headlineMedium, color = CanzukBlue)
            Spacer(Modifier.height(8.dp))
            Text("Visual reasoning", style = MaterialTheme.typography.bodyMedium, color = TextDim)
            Spacer(Modifier.height(4.dp))
            Text("DAG visualisation, derivation trees, diagram generation", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
        }
    }
}
