package com.enrique.ai

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            EnriqueAIScreen()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EnriqueAIScreen(viewModel: ChatViewModel = viewModel()) {
    val context = LocalContext.current
    val messages = viewModel.messages
    val tokenCount = viewModel.tokenCount
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    var showSidebar by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        viewModel.initialize(context)
    }

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    val darkBackground = Color(0xFF0A0A0F)
    val darkSurface = Color(0xFF111118)
    val cardColor = Color(0xFF181825)
    val card2Color = Color(0xFF20202E)
    val primaryColor = Color(0xFF10A37F)
    val textColor = Color(0xFFE4E4EC)
    val text2Color = Color(0xFFC8C8D4)
    val mutedColor = Color(0xFF7D7D90)

    Scaffold(
        modifier = Modifier.background(darkBackground),
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("ENRIQUE NN", color = textColor, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                        Spacer(Modifier.width(8.dp))
                        Text("v3.0", color = mutedColor, fontSize = 10.sp)
                    }
                },
                actions = {
                    Text("⚡ $tokenCount", color = mutedColor, fontSize = 12.sp)
                    Spacer(Modifier.width(8.dp))
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = darkSurface),
                navigationIcon = {
                    IconButton(onClick = { showSidebar = !showSidebar }) {
                        Text("=", color = mutedColor, fontSize = 18.sp)
                    }
                }
            )
        },
        containerColor = darkBackground
    ) { padding ->
        Row(Modifier.fillMaxSize().padding(padding)) {
            // Sidebar
            if (showSidebar) {
                Column(
                    modifier = Modifier
                        .width(260.dp)
                        .fillMaxHeight()
                        .background(darkSurface)
                        .padding(12.dp)
                ) {
                    Text("ENRIQUE AI", color = textColor, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                    Spacer(Modifier.height(16.dp))

                    Button(
                        onClick = { viewModel.newChat() },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = cardColor),
                        shape = RoundedCornerShape(8.dp)
                    ) { Text("+ New Chat", color = textColor, fontSize = 12.sp) }

                    Spacer(Modifier.height(8.dp))

                    Button(
                        onClick = {
                            viewModel.trainNetwork(context)
                            Toast.makeText(context, "Training complete!", Toast.LENGTH_SHORT).show()
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = primaryColor),
                        shape = RoundedCornerShape(8.dp)
                    ) { Text("🧠 Train Network", color = Color.White, fontSize = 12.sp) }

                    Spacer(Modifier.height(16.dp))
                    Text("STATS", color = mutedColor, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(8.dp))
                    Text("Status: ${if (viewModel.isTrained) "Trained" else "Fresh"}", color = text2Color, fontSize = 12.sp)
                    Text("Messages: ${messages.size}", color = text2Color, fontSize = 12.sp)
                    Text("Tokens: $tokenCount", color = text2Color, fontSize = 12.sp)
                }
            }

            // Main chat area
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .background(darkBackground)
            ) {
                // Messages
                if (messages.isEmpty()) {
                    Column(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth()
                            .padding(32.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Text(
                            "E",
                            fontSize = 48.sp,
                            fontWeight = FontWeight.Bold,
                            color = primaryColor
                        )
                        Spacer(Modifier.height(12.dp))
                        Text(
                            "ENRIQUE NN",
                            fontSize = 22.sp,
                            fontWeight = FontWeight.Bold,
                            color = textColor
                        )
                        Spacer(Modifier.height(8.dp))
                        Text(
                            "Neural Network from Scratch • LSTM",
                            fontSize = 13.sp,
                            color = mutedColor,
                            textAlign = TextAlign.Center
                        )
                        Spacer(Modifier.height(24.dp))

                        // Quick action chips
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            listOf("Hello", "Python", "Website").forEach { text ->
                                Surface(
                                    onClick = {
                                        inputText = text
                                        viewModel.sendMessage(context, text)
                                    },
                                    shape = RoundedCornerShape(10.dp),
                                    color = cardColor
                                ) {
                                    Text(text, modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                                        color = mutedColor, fontSize = 12.sp)
                                }
                            }
                        }
                        Spacer(Modifier.height(12.dp))
                        Text(
                            "Click Train Network in sidebar to start",
                            fontSize = 10.sp, color = mutedColor.copy(alpha = 0.6f)
                        )
                    }
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth()
                            .padding(horizontal = 14.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                        contentPadding = PaddingValues(vertical = 12.dp)
                    ) {
                        items(messages) { msg ->
                            val isUser = msg.role == "user"
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
                            ) {
                                Column(
                                    modifier = Modifier
                                        .widthIn(max = 300.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(if (isUser) cardColor else card2Color)
                                        .padding(12.dp)
                                ) {
                                    Text(
                                        msg.content,
                                        color = text2Color,
                                        fontSize = 14.sp,
                                        lineHeight = 22.sp
                                    )
                                }
                            }
                        }
                    }
                }

                // Input
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(darkSurface)
                        .padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        modifier = Modifier.weight(1f),
                        placeholder = {
                            Text(if (viewModel.isTrained) "Type a message..." else "Train the network first!",
                                color = mutedColor.copy(alpha = 0.5f), fontSize = 14.sp)
                        },
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedTextColor = textColor,
                            unfocusedTextColor = textColor,
                            cursorColor = primaryColor,
                            focusedBorderColor = primaryColor,
                            unfocusedBorderColor = Color(0xFF2A2A3A),
                            focusedContainerColor = cardColor,
                            unfocusedContainerColor = cardColor
                        ),
                        shape = RoundedCornerShape(12.dp),
                        singleLine = true,
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                        keyboardActions = KeyboardActions(
                            onSend = {
                                viewModel.sendMessage(context, inputText)
                                inputText = ""
                            }
                        )
                    )
                    Spacer(Modifier.width(8.dp))
                    Button(
                        onClick = {
                            viewModel.sendMessage(context, inputText)
                            inputText = ""
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = primaryColor),
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.size(48.dp),
                        contentPadding = PaddingValues(0.dp)
                    ) {
                        Text(">", color = Color.White, fontSize = 18.sp)
                    }
                }
            }
        }
    }
}
