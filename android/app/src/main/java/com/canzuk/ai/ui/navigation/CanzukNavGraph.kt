package com.canzuk.ai.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.canzuk.ai.ui.screens.CanvasScreen
import com.canzuk.ai.ui.screens.CodeScreen
import com.canzuk.ai.ui.screens.DocumentScreen
import com.canzuk.ai.ui.screens.MoreScreen

enum class Screen(val route: String) {
    Chat("chat"),
    Document("document"),
    Code("code"),
    Canvas("canvas"),
    More("more"),
}

@Composable
fun CanzukNavGraph(navController: NavHostController) {
    NavHost(navController = navController, startDestination = Screen.Chat.route) {
        composable(Screen.Chat.route) { /* ChatScreen — wired in integration */ }
        composable(Screen.Document.route) { DocumentScreen() }
        composable(Screen.Code.route) { CodeScreen() }
        composable(Screen.Canvas.route) { CanvasScreen() }
        composable(Screen.More.route) { MoreScreen() }
    }
}
