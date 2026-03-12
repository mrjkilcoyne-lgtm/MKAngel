package com.canzuk.ai.ui.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Brush
import androidx.compose.material.icons.outlined.ChatBubbleOutline
import androidx.compose.material.icons.outlined.Code
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.MoreHoriz
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavController
import androidx.navigation.compose.currentBackStackEntryAsState
import com.canzuk.ai.ui.theme.CanzukRed
import com.canzuk.ai.ui.theme.TextDim

data class BottomNavItem(
    val screen: Screen,
    val label: String,
    val icon: ImageVector,
)

val bottomNavItems = listOf(
    BottomNavItem(Screen.Chat, "Chat", Icons.Outlined.ChatBubbleOutline),
    BottomNavItem(Screen.Document, "Document", Icons.Outlined.Description),
    BottomNavItem(Screen.Code, "Code", Icons.Outlined.Code),
    BottomNavItem(Screen.Canvas, "Canvas", Icons.Outlined.Brush),
    BottomNavItem(Screen.More, "More", Icons.Outlined.MoreHoriz),
)

@Composable
fun CanzukBottomBar(navController: NavController) {
    val currentRoute = navController.currentBackStackEntryAsState().value?.destination?.route

    NavigationBar(
        containerColor = MaterialTheme.colorScheme.background,
    ) {
        bottomNavItems.forEach { item ->
            NavigationBarItem(
                selected = currentRoute == item.screen.route,
                onClick = {
                    navController.navigate(item.screen.route) {
                        popUpTo(Screen.Chat.route) { saveState = true }
                        launchSingleTop = true
                        restoreState = true
                    }
                },
                label = {
                    Text(
                        item.label,
                        style = MaterialTheme.typography.labelSmall,
                    )
                },
                icon = { Icon(item.icon, contentDescription = item.label) },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = CanzukRed,
                    selectedTextColor = CanzukRed,
                    unselectedIconColor = TextDim,
                    unselectedTextColor = TextDim,
                    indicatorColor = CanzukRed.copy(alpha = 0.1f),
                ),
            )
        }
    }
}
