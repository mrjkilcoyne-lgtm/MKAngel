package com.canzuk.ai.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.canzuk.ai.ui.theme.*

@Composable
fun CodeScreen() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Code", style = MaterialTheme.typography.headlineMedium, color = CanzukBlue)
            Spacer(Modifier.height(8.dp))
            Text("Grammar-driven code generation", style = MaterialTheme.typography.bodyMedium, color = TextDim)
            Spacer(Modifier.height(4.dp))
            Text("Computational grammar domain — write, explain, refactor", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
        }
    }
}
