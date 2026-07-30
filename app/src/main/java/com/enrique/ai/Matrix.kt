package com.enrique.ai

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

class Matrix(val rows: Int, val cols: Int) {
    val data: FloatArray = FloatArray(rows * cols)

    init {
        val scale = sqrt(2.0f / (rows + cols))
        for (i in data.indices) {
            data[i] = (Random.nextFloat() * 2 - 1) * scale
        }
    }

    constructor(rows: Int, cols: Int, values: FloatArray) : this(rows, cols) {
        System.arraycopy(values, 0, data, 0, minOf(values.size, data.size))
    }

    operator fun get(r: Int, c: Int): Float = data[r * cols + c]
    operator fun set(r: Int, c: Int, v: Float) { data[r * cols + c] = v }

    operator fun times(other: Matrix): Matrix {
        require(cols == other.rows) { "Matrix dims: ${rows}x$cols * ${other.rows}x${other.cols}" }
        val out = Matrix(rows, other.cols)
        for (i in 0 until rows) {
            for (j in 0 until other.cols) {
                var sum = 0f
                for (k in 0 until cols) sum += this[i, k] * other[k, j]
                out[i, j] = sum
            }
        }
        return out
    }

    operator fun plus(other: Matrix): Matrix {
        require(rows == other.rows && cols == other.cols)
        val out = Matrix(rows, cols)
        for (i in data.indices) out.data[i] = data[i] + other.data[i]
        return out
    }

    operator fun plus(scalar: Float): Matrix {
        val out = Matrix(rows, cols)
        for (i in data.indices) out.data[i] = data[i] + scalar
        return out
    }

    fun elmul(other: Matrix): Matrix {
        val out = Matrix(rows, cols)
        for (i in data.indices) out.data[i] = data[i] * other.data[i]
        return out
    }

    fun map(f: (Float) -> Float): Matrix {
        val out = Matrix(rows, cols)
        for (i in data.indices) out.data[i] = f(data[i])
        return out
    }

    fun scale(s: Float): Matrix {
        val out = Matrix(rows, cols)
        for (i in data.indices) out.data[i] = data[i] * s
        return out
    }

    fun clip(maxVal: Float) {
        for (i in data.indices) data[i] = max(-maxVal, min(maxVal, data[i]))
    }

    fun copy(): Matrix {
        val out = Matrix(rows, cols)
        System.arraycopy(data, 0, out.data, 0, data.size)
        return out
    }

    fun T(): Matrix {
        val out = Matrix(cols, rows)
        for (i in 0 until rows) for (j in 0 until cols) out[j, i] = this[i, j]
        return out
    }

    fun toFloatArray(): FloatArray = data.copyOf()

    companion object {
        fun sig(x: Float): Float = 1f / (1f + exp(-max(-50f, min(50f, x))))
        fun dsig(y: Float): Float = y * (1 - y)
        fun tanh(x: Float): Float = kotlin.math.tanh(x)
        fun dtanh(y: Float): Float = 1 - y * y

        fun softmax(x: FloatArray): FloatArray {
            val max = x.max()
            val exps = x.map { exp(it - max) }
            val sum = exps.sum()
            return exps.map { it / sum }.toFloatArray()
        }

        fun randomMatrix(rows: Int, cols: Int, scale: Float = 1f): Matrix {
            val m = Matrix(rows, cols)
            val s = sqrt(2.0f / (rows + cols)) * scale
            for (i in m.data.indices) m.data[i] = (Random.nextFloat() * 2 - 1) * s
            return m
        }

        fun zeros(rows: Int, cols: Int): Matrix {
            val m = Matrix(rows, cols)
            m.data.fill(0f)
            return m
        }
    }
}
