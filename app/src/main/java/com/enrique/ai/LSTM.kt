package com.enrique.ai

class LSTMCell(val inputSize: Int, val hiddenSize: Int) {
    lateinit var Wf: Matrix; lateinit var Uf: Matrix; lateinit var bf: Matrix
    lateinit var Wi: Matrix; lateinit var Ui: Matrix; lateinit var bi: Matrix
    lateinit var Wc: Matrix; lateinit var Uc: Matrix; lateinit var bc: Matrix
    lateinit var Wo: Matrix; lateinit var Uo: Matrix; lateinit var bo: Matrix

    val params: List<Matrix> get() = listOf(Wf, Uf, bf, Wi, Ui, bi, Wc, Uc, bc, Wo, Uo, bo)

    init {
        Wf = Matrix.randomMatrix(inputSize, hiddenSize)
        Uf = Matrix.randomMatrix(hiddenSize, hiddenSize)
        bf = Matrix.zeros(1, hiddenSize)
        Wi = Matrix.randomMatrix(inputSize, hiddenSize)
        Ui = Matrix.randomMatrix(hiddenSize, hiddenSize)
        bi = Matrix.zeros(1, hiddenSize)
        Wc = Matrix.randomMatrix(inputSize, hiddenSize)
        Uc = Matrix.randomMatrix(hiddenSize, hiddenSize)
        bc = Matrix.zeros(1, hiddenSize)
        Wo = Matrix.randomMatrix(inputSize, hiddenSize)
        Uo = Matrix.randomMatrix(hiddenSize, hiddenSize)
        bo = Matrix.zeros(1, hiddenSize)
    }

    data class FwdResult(
        val h: FloatArray, val c: FloatArray,
        val f: FloatArray, val i: FloatArray,
        val cc: FloatArray, val o: FloatArray
    )

    fun forward(x: FloatArray, hPrev: FloatArray, cPrev: FloatArray): FwdResult {
        val xm = Matrix(1, inputSize).apply { x.copyInto(data) }
        val hm = Matrix(1, hiddenSize).apply { hPrev.copyInto(data) }

        val f = sig(xm * Wf + hm * Uf + bf)
        val i = sig(xm * Wi + hm * Ui + bi)
        val cc = tanh(xm * Wc + hm * Uc + bc)
        val o = sig(xm * Wo + hm * Uo + bo)

        val c = FloatArray(hiddenSize)
        val h = FloatArray(hiddenSize)
        for (j in 0 until hiddenSize) {
            c[j] = f.data[j] * cPrev[j] + i.data[j] * cc.data[j]
            h[j] = o.data[j] * Matrix.tanh(c[j])
        }
        return FwdResult(h, c, f.data, i.data, cc.data, o.data)
    }

    private fun sig(m: Matrix): Matrix = m.map { Matrix.sig(it) }
    private fun tanh(m: Matrix): Matrix = m.map { Matrix.tanh(it) }
}

class ENRIQUELSTM(
    val vocabSize: Int,
    val hiddenSize: Int,
    val embedSize: Int,
    val numLayers: Int = 2
) {
    val Wex = Matrix.randomMatrix(vocabSize, embedSize)
    val layers = Array(numLayers) { LSTMCell(if (it == 0) embedSize else hiddenSize, hiddenSize) }
    val Why = Matrix.randomMatrix(hiddenSize, vocabSize)
    val by = Matrix.zeros(1, vocabSize)

    var iter = 0
    var smoothLoss = 10f

    fun forward(inputIdx: Int, hStates: Array<FloatArray>, cStates: Array<FloatArray>): FwdResult {
        val x = Matrix(1, embedSize).apply {
            for (j in 0 until embedSize) data[j] = Wex[inputIdx, j]
        }

        val newH = Array(numLayers) { FloatArray(hiddenSize) }
        val newC = Array(numLayers) { FloatArray(hiddenSize) }

        var currentInput = x.data
        for (l in 0 until numLayers) {
            val result = layers[l].forward(
                currentInput,
                hStates[l],
                cStates[l]
            )
            newH[l] = result.h
            newC[l] = result.c
            if (l == 0) {
                val hm = Matrix(1, hiddenSize).apply { result.h.copyInto(data) }
                currentInput = hm.data
            } else {
                currentInput = newH[l]
            }
        }

        val yRaw = (Matrix(1, hiddenSize).apply { newH.last().copyInto(data) } * Why) + by
        val yProbs = Matrix.softmax(yRaw.data)

        return FwdResult(newH, newC, yProbs)
    }

    data class FwdResult(
        val h: Array<FloatArray>,
        val c: Array<FloatArray>,
        val yProbs: FloatArray
    )

    fun sample(yProbs: FloatArray, temperature: Float): Int {
        val scaled = yProbs.map { p -> kotlin.math.pow(p, 1f / temperature) }
        val sum = scaled.sum()
        val normed = scaled.map { it / sum }
        var r = kotlin.random.Random.nextFloat()
        for (i in normed.indices) {
            r -= normed[i]
            if (r <= 0) return i
        }
        return normed.size - 1
    }

    fun generate(
        chars: List<Char>,
        charToIdx: Map<Char, Int>,
        prompt: String,
        maxLen: Int = 200,
        temperature: Float = 0.7f
    ): String {
        var output = prompt
        var hStates = Array(numLayers) { FloatArray(hiddenSize) }
        var cStates = Array(numLayers) { FloatArray(hiddenSize) }

        for (ch in output) {
            val ix = charToIdx[ch] ?: 0
            val fwd = forward(ix, hStates, cStates)
            hStates = fwd.h; cStates = fwd.c
        }

        for (i in 0 until maxLen) {
            val last = output.lastOrNull() ?: ' '
            val ix = charToIdx[last] ?: 0
            val fwd = forward(ix, hStates, cStates)
            hStates = fwd.h; cStates = fwd.c
            val nextIx = sample(fwd.yProbs, temperature)
            output += chars.getOrElse(nextIx) { '?' }
            if (chars.getOrElse(nextIx) { ' ' } == '.' ||
                chars.getOrElse(nextIx) { ' ' } == '。') break
        }
        return output
    }

    fun train(seq: String, charToIdx: Map<Char, Int>, lr: Float = 0.002f) {
        if (seq.length < 5) return
        val indices = seq.map { charToIdx[it] ?: 0 }

        var hStates = Array(numLayers) { FloatArray(hiddenSize) }
        var cStates = Array(numLayers) { FloatArray(hiddenSize) }

        data class Step(val h: Array<FloatArray>, val c: Array<FloatArray>, val y: FloatArray, val ix: Int, val tgt: Int)

        val steps = mutableListOf<Step>()
        var loss = 0f

        for (t in 0 until indices.size - 1) {
            val fwd = forward(indices[t], hStates, cStates)
            hStates = fwd.h; cStates = fwd.c
            loss += -kotlin.math.log(maxOf(fwd.yProbs[indices[t + 1]], 1e-10f))
            steps.add(Step(
                hStates.map { it.copyOf() }.toTypedArray(),
                cStates.map { it.copyOf() }.toTypedArray(),
                fwd.yProbs.copyOf(), indices[t], indices[t + 1]
            ))
        }

        smoothLoss = smoothLoss * 0.999f + (loss / steps.size) * 0.001f

        for (step in steps) {
            for (j in 0 until vocabSize) {
                var grad = step.y[j] - if (j == step.tgt) 1f else 0f
                if (kotlin.math.abs(grad) > 0.05f) {
                    grad = maxOf(-5f, minOf(5f, grad)) * 0.1f
                    for (k in 0 until embedSize) {
                        Wex.data[step.ix * embedSize + k] -= lr * grad
                    }
                    for (k in 0 until hiddenSize) {
                        Why.data[k * vocabSize + j] -= lr * grad * step.h.last()[k] * 0.1f
                        by.data[j] -= lr * grad * 0.1f
                    }
                }
            }
        }
        iter++
    }

    fun save(): Map<String, Any> {
        return mapOf(
            "vocabSize" to vocabSize,
            "hiddenSize" to hiddenSize,
            "embedSize" to embedSize,
            "numLayers" to numLayers,
            "iter" to iter,
            "loss" to smoothLoss,
            "Wex" to Wex.data.toList(),
            "layers" to layers.map { layer ->
                layer.params.map { it.data.toList() }
            },
            "Why" to Why.data.toList(),
            "by" to by.data.toList()
        )
    }

    fun load(data: Map<String, Any>) {
        @Suppress("UNCHECKED_CAST")
        val wexList = data["Wex"] as List<Number>
        wexList.forEachIndexed { i, v -> Wex.data[i] = v.toFloat() }

        val layersData = data["layers"] as List<List<List<Number>>>
        layersData.forEachIndexed { l, layerData ->
            layerData.forEachIndexed { p, paramData ->
                if (l < layers.size && p < layers[l].params.size) {
                    paramData.forEachIndexed { i, v ->
                        layers[l].params[p].data[i] = v.toFloat()
                    }
                }
            }
        }

        val whyList = data["Why"] as List<Number>
        whyList.forEachIndexed { i, v -> Why.data[i] = v.toFloat() }

        val byList = data["by"] as List<Number>
        byList.forEachIndexed { i, v -> by.data[i] = v.toFloat() }

        iter = (data["iter"] as? Number)?.toInt() ?: 0
        smoothLoss = (data["loss"] as? Number)?.toFloat() ?: 10f
    }
}
