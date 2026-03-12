package com.canzuk.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import androidx.navigation.compose.rememberNavController
import com.canzuk.ai.ui.navigation.CanzukBottomBar
import com.canzuk.ai.ui.navigation.CanzukNavGraph
import com.canzuk.ai.ui.theme.CanzukTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            CanzukTheme {
                val navController = rememberNavController()
                Scaffold(
                    bottomBar = { CanzukBottomBar(navController) },
                ) { padding ->
                    Box(modifier = Modifier.padding(padding)) {
                        CanzukNavGraph(navController)
                    }
                }
            }
        }
    }
}
