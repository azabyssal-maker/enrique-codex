package com.enrique.ai

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import org.json.JSONArray
import org.json.JSONObject

data class Message(val role: String, val content: String, val time: Long = System.currentTimeMillis())

class ChatViewModel : ViewModel() {
    var messages = mutableStateListOf<Message>()
    var tokenCount by mutableIntStateOf(0)
    var isBusy = false
    var isTrained = false

    private var nn: ENRIQUELSTM? = null
    private var vocab: Vocab? = null
    private var trainingData = mutableListOf<String>()

    fun initialize(context: Context) {
        vocab = Vocab.default()
        val saved = loadModel(context)
        if (saved != null) {
            nn = ENRIQUELSTM(
                saved["vocabSize"] as Int,
                saved["hiddenSize"] as Int,
                saved["embedSize"] as Int,
                (saved["numLayers"] as? Int) ?: 2
            )
            nn!!.load(saved)
            isTrained = nn!!.iter > 0
        } else {
            nn = ENRIQUELSTM(vocab!!.size, 256, 64, 2)
        }
        loadMessages(context)
    }

    fun sendMessage(context: Context, text: String) {
        if (text.isBlank() || isBusy) return
        isBusy = true
        messages.add(Message("user", text))
        tokenCount += text.length

        val reply = generateResponse(text)
        messages.add(Message("ai", reply))
        tokenCount += reply.length

        saveMessages(context)
        saveModel(context)
        isBusy = false
    }

    private fun generateResponse(msg: String): String {
        val nn = nn ?: return "请先在侧边栏训练神经网络！"
        val vocab = vocab ?: return "词汇表未初始化"

        if (nn.iter < 3) {
            return fallbackResponse(msg)
        }

        try {
            var reply = nn.generate(vocab.chars, vocab.charToIdx, msg, 200, 0.7f)
            reply = reply.removePrefix(msg).trim()
            if (reply.length > 300) reply = reply.substring(0, 300)
            val stopIdx = maxOf(reply.lastIndexOf('。'), reply.lastIndexOf('.'))
            if (stopIdx > 10) reply = reply.substring(0, stopIdx + 1)
            return reply.ifEmpty { "..." }
        } catch (e: Exception) {
            return "生成回复时出错: ${e.message}"
        }
    }

    private fun fallbackResponse(msg: String): String {
        val m = msg.lowercase().trim()
        return when {
            m.startsWith("hi") || m.startsWith("hello") || m.startsWith("你好") ->
                "你好！我是 ENRIQUE AI。请点击"训练网络"按钮训练我的神经网络！"
            m.contains("python") -> "Python 是一种高级编程语言。训练后我能写 Python 代码！"
            else -> "我正在学习！请点击「训练网络」按钮训练我。训练后我能更好地回答你的问题！"
        }
    }

    fun trainNetwork(context: Context) {
        val vocab = vocab ?: return
        val allText = PRETRAIN_DATA.joinToString(" ")
        nn = ENRIQUELSTM(vocab.size, 256, 64, 2)
        this.vocab = Vocab.build(listOf(allText) + PRETRAIN_DATA)

        for (epoch in 0 until 30) {
            nn!!.train(allText, this.vocab!!.charToIdx, 0.005f)
            if (nn!!.smoothLoss.isNaN() || nn!!.smoothLoss > 100f) break
        }
        isTrained = true
        saveModel(context)
    }

    fun newChat() {
        messages.clear()
        tokenCount = 0
    }

    fun saveMessages(context: Context) {
        try {
            val arr = JSONArray()
            for (msg in messages) {
                val obj = JSONObject()
                obj.put("role", msg.role)
                obj.put("content", msg.content)
                obj.put("time", msg.time)
                arr.put(obj)
            }
            val prefs = context.getSharedPreferences("enrique", Context.MODE_PRIVATE)
            prefs.edit().putString("messages", arr.toString())
                .putInt("tokens", tokenCount).apply()
        } catch (_: Exception) {}
    }

    fun loadMessages(context: Context) {
        try {
            val prefs = context.getSharedPreferences("enrique", Context.MODE_PRIVATE)
            val json = prefs.getString("messages", null) ?: return
            tokenCount = prefs.getInt("tokens", 0)
            val arr = JSONArray(json)
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                messages.add(Message(
                    obj.getString("role"),
                    obj.getString("content"),
                    obj.getLong("time")
                ))
            }
        } catch (_: Exception) {}
    }

    fun saveModel(context: Context) {
        try {
            val prefs = context.getSharedPreferences("enrique", Context.MODE_PRIVATE)
            val data = nn?.save() ?: return
            val json = JSONObject()
            for ((key, value) in data) {
                when (value) {
                    is Int -> json.put(key, value)
                    is Float -> json.put(key, value.toDouble())
                    is List<*> -> json.put(key, JSONArray(value))
                    else -> json.put(key, value)
                }
            }
            prefs.edit().putString("model", json.toString()).apply()
        } catch (_: Exception) {}
    }

    fun loadModel(context: Context): Map<String, Any>? {
        return try {
            val prefs = context.getSharedPreferences("enrique", Context.MODE_PRIVATE)
            val json = prefs.getString("model", null) ?: return null
            val obj = JSONObject(json)
            val result = mutableMapOf<String, Any>()
            for (key in obj.keys()) {
                val value = obj.get(key)
                result[key] = when (value) {
                    is Number -> value
                    is JSONArray -> {
                        val list = mutableListOf<Any>()
                        for (i in 0 until value.length()) {
                            val v = value.get(i)
                            if (v is JSONArray) {
                                val inner = mutableListOf<Any>()
                                for (j in 0 until v.length()) inner.add(v.get(j))
                                list.add(inner)
                            } else {
                                list.add(v)
                            }
                        }
                        list
                    }
                    else -> value
                }
            }
            result
        } catch (_: Exception) { null }
    }

    companion object {
        val PRETRAIN_DATA = listOf(
            "Hello! I am ENRIQUE AI. I am a neural network built from scratch in Kotlin.",
            "I can help you write code, create websites, and answer questions.",
            "你好！我是 ENRIQUE AI。我是一个从零构建的神经网络。",
            "Machine learning is about teaching computers to learn from data.",
            "Neural networks are inspired by the human brain.",
            "I can speak Chinese and English.",
            "ENRIQUE AI learns from every conversation. The more we talk the smarter I become.",
            "Coding is fun. Practice makes perfect.",
            "JavaScript and Kotlin are both great programming languages.",
            "Python is good for data science and machine learning."
        )
    }
}

data class Vocab(
    val chars: List<Char>,
    val charToIdx: Map<Char, Int>,
    val idxToChar: Map<Int, Char>,
    val size: Int
) {
    companion object {
        fun default(): Vocab {
            val allChars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-()[]{}@#\$%^&*+=/\\|~`<>的一是不了人我在有他这中大来上国个到说们为和地出要时看没只如开会可对生年发成部民能方进同经法面又分前平定后然间因长期把多公从里事好她些还其动两动手已外小高点长当都水化得自她什但者于些又她你日"
            val unique = allChars.toSet().toList().sorted()
            val c2i = unique.mapIndexed { i, c -> c to i }.toMap()
            val i2c = unique.mapIndexed { i, c -> i to c }.toMap()
            return Vocab(unique, c2i, i2c, unique.size)
        }

        fun build(texts: List<String>): Vocab {
            val all = texts.flatMap { it.toList() }.toSet().toList().sorted()
            val c2i = all.mapIndexed { i, c -> c to i }.toMap()
            val i2c = all.mapIndexed { i, c -> i to c }.toMap()
            return Vocab(all, c2i, i2c, all.size)
        }
    }
}
