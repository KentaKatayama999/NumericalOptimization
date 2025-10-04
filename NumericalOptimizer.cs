using System;
using SysVector3 = System.Numerics.Vector3;
using MathNet.Numerics.LinearAlgebra;

namespace NumericalOptimization
{
    /// <summary>
    /// 数値最適化のためのユーティリティクラス
    /// </summary>
    public static class NumericalOptimizer
    {
        /// <summary>
        /// ヤコビアン行列を数値微分で計算
        /// </summary>
        /// <param name="point">評価点</param>
        /// <param name="function">ベクトル値関数 f: R^n → R^m</param>
        /// <param name="stepSize">微分のステップサイズ</param>
        /// <returns>ヤコビアン行列 (m × n)</returns>
        public static Matrix<double> ComputeJacobian(
            Vector<double> point,
            Func<Vector<double>, Vector<double>> function,
            double stepSize = 1e-6)
        {
            if (point == null)
                throw new ArgumentNullException(nameof(point));
            if (function == null)
                throw new ArgumentNullException(nameof(function));
            if (stepSize <= 0)
                throw new ArgumentException("ステップサイズは正の値である必要があります", nameof(stepSize));

            int n = point.Count; // 入力次元
            var f0 = function(point);
            int m = f0.Count; // 出力次元

            var jacobian = Matrix<double>.Build.Dense(m, n);

            // 各列を数値微分で計算
            for (int j = 0; j < n; j++)
            {
                var pointPlusH = point.Clone();
                pointPlusH[j] += stepSize;

                var fPlusH = function(pointPlusH);
                var df = (fPlusH - f0) / stepSize;

                for (int i = 0; i < m; i++)
                {
                    jacobian[i, j] = df[i];
                }
            }

            return jacobian;
        }

        /// <summary>
        /// スカラー関数の勾配ベクトルを数値微分で計算
        /// </summary>
        /// <param name="point">評価点</param>
        /// <param name="function">スカラー関数 f: R^n → R</param>
        /// <param name="stepSize">微分のステップサイズ</param>
        /// <returns>勾配ベクトル</returns>
        public static Vector<double> ComputeGradient(
            Vector<double> point,
            Func<Vector<double>, double> function,
            double stepSize = 1e-6)
        {
            if (point == null)
                throw new ArgumentNullException(nameof(point));
            if (function == null)
                throw new ArgumentNullException(nameof(function));
            if (stepSize <= 0)
                throw new ArgumentException("ステップサイズは正の値である必要があります", nameof(stepSize));

            int n = point.Count;
            var gradient = Vector<double>.Build.Dense(n);
            var f0 = function(point);

            for (int i = 0; i < n; i++)
            {
                var pointPlusH = point.Clone();
                pointPlusH[i] += stepSize;

                var fPlusH = function(pointPlusH);
                gradient[i] = (fPlusH - f0) / stepSize;
            }

            return gradient;
        }

        /// <summary>
        /// Levenberg-Marquardt法の1ステップを実行
        /// </summary>
        /// <param name="currentPoint">現在の点</param>
        /// <param name="residualFunction">残差関数 r: R^n → R^m</param>
        /// <param name="lambda">ダンピングパラメータ（大きいほど勾配降下法に近づく）</param>
        /// <param name="stepSize">ヤコビアン計算時の微分ステップサイズ</param>
        /// <returns>更新された点と残差のノルム</returns>
        public static (Vector<double> newPoint, double residualNorm) LevenbergMarquardtStep(
            Vector<double> currentPoint,
            Func<Vector<double>, Vector<double>> residualFunction,
            double lambda = 0.01,
            double stepSize = 1e-6)
        {
            if (currentPoint == null)
                throw new ArgumentNullException(nameof(currentPoint));
            if (residualFunction == null)
                throw new ArgumentNullException(nameof(residualFunction));
            if (lambda <= 0)
                throw new ArgumentException("ダンピングパラメータは正の値である必要があります", nameof(lambda));

            // 残差とヤコビアンを計算
            var r = residualFunction(currentPoint);
            var J = ComputeJacobian(currentPoint, residualFunction, stepSize);

            // J^T * J + λI を計算
            var JtJ = J.TransposeThisAndMultiply(J);
            var I = Matrix<double>.Build.DenseIdentity(currentPoint.Count);
            var A = JtJ + lambda * I;

            // J^T * r を計算
            var Jtr = J.Transpose() * r;

            // (J^T * J + λI) * δ = -J^T * r を解く
            Vector<double> delta;
            try
            {
                delta = A.Solve(-Jtr);
            }
            catch (Exception)
            {
                // 特異行列の場合は更新なし
                return (currentPoint.Clone(), r.L2Norm());
            }

            // 点を更新
            var newPoint = currentPoint + delta;

            // 残差のノルムを計算
            var newResidual = residualFunction(newPoint);
            var residualNorm = newResidual.L2Norm();

            return (newPoint, residualNorm);
        }

        /// <summary>
        /// Levenberg-Marquardt法を完全に実行（収束まで反復）
        /// </summary>
        /// <param name="initialPoint">初期点</param>
        /// <param name="residualFunction">残差関数</param>
        /// <param name="tolerance">収束判定の許容誤差</param>
        /// <param name="maxIterations">最大反復回数</param>
        /// <param name="initialLambda">初期ダンピングパラメータ</param>
        /// <param name="lambdaFactor">ダンピングパラメータの調整係数</param>
        /// <returns>最適化結果</returns>
        public static OptimizationResult LevenbergMarquardt(
            Vector<double> initialPoint,
            Func<Vector<double>, Vector<double>> residualFunction,
            double tolerance = 1e-6,
            int maxIterations = 100,
            double initialLambda = 0.01,
            double lambdaFactor = 10.0)
        {
            if (initialPoint == null)
                throw new ArgumentNullException(nameof(initialPoint));
            if (residualFunction == null)
                throw new ArgumentNullException(nameof(residualFunction));

            var currentPoint = initialPoint.Clone();
            var currentResidual = residualFunction(currentPoint);
            var currentNorm = currentResidual.L2Norm();
            double lambda = initialLambda;

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                // L-Mステップを実行
                var (newPoint, newNorm) = LevenbergMarquardtStep(currentPoint, residualFunction, lambda);

                // 改善があったか確認
                if (newNorm < currentNorm)
                {
                    // 改善あり：点を更新してλを減らす
                    currentPoint = newPoint;
                    currentNorm = newNorm;
                    lambda /= lambdaFactor;

                    // 収束判定
                    if (currentNorm < tolerance)
                    {
                        return new OptimizationResult
                        {
                            Success = true,
                            OptimalPoint = currentPoint,
                            ResidualNorm = currentNorm,
                            Iterations = iteration + 1
                        };
                    }
                }
                else
                {
                    // 改善なし：λを増やす
                    lambda *= lambdaFactor;
                }
            }

            // 最大反復回数に達した
            return new OptimizationResult
            {
                Success = false,
                OptimalPoint = currentPoint,
                ResidualNorm = currentNorm,
                Iterations = maxIterations
            };
        }

        /// <summary>
        /// Vector3からMathNet.Numericsのベクトルに変換
        /// </summary>
        public static Vector<double> ToMathNetVector(SysVector3 v)
        {
            var result = Vector<double>.Build.Dense(3);
            result[0] = v.X;
            result[1] = v.Y;
            result[2] = v.Z;
            return result;
        }

        /// <summary>
        /// Vector3からMathNet.Numericsのベクトルに変換（2D、Z=0を無視）
        /// </summary>
        public static Vector<double> ToMathNetVector2D(SysVector3 v)
        {
            var result = Vector<double>.Build.Dense(2);
            result[0] = v.X;
            result[1] = v.Y;
            return result;
        }

        /// <summary>
        /// MathNet.NumericsのベクトルからVector3に変換
        /// </summary>
        public static SysVector3 ToVector3(Vector<double> v)
        {
            if (v.Count == 2)
                return new SysVector3((float)v[0], (float)v[1], 0);
            else if (v.Count == 3)
                return new SysVector3((float)v[0], (float)v[1], (float)v[2]);
            else
                throw new ArgumentException("ベクトルの次元は2または3である必要があります");
        }

        /// <summary>
        /// ニュートン法による1変数方程式の求解（数値微分版）
        /// </summary>
        /// <param name="initialGuess">初期推定値</param>
        /// <param name="function">方程式 f(x) = 0 の左辺</param>
        /// <param name="tolerance">収束判定の許容誤差</param>
        /// <param name="maxIterations">最大反復回数</param>
        /// <param name="stepSize">数値微分のステップサイズ</param>
        /// <returns>求解結果</returns>
        public static NewtonResult Newton(
            double initialGuess,
            Func<double, double> function,
            double tolerance = 1e-6,
            int maxIterations = 100,
            double stepSize = 1e-6)
        {
            if (function == null)
                throw new ArgumentNullException(nameof(function));
            if (tolerance <= 0)
                throw new ArgumentException("許容誤差は正の値である必要があります", nameof(tolerance));
            if (stepSize <= 0)
                throw new ArgumentException("ステップサイズは正の値である必要があります", nameof(stepSize));

            double x = initialGuess;

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                double fx = function(x);

                // 収束判定
                if (Math.Abs(fx) < tolerance)
                {
                    return new NewtonResult
                    {
                        Success = true,
                        Root = x,
                        FunctionValue = fx,
                        Iterations = iteration + 1
                    };
                }

                // 数値微分で導関数を計算
                double fxPlusH = function(x + stepSize);
                double derivative = (fxPlusH - fx) / stepSize;

                // 導関数が0に近い場合は失敗
                if (Math.Abs(derivative) < 1e-12)
                {
                    return new NewtonResult
                    {
                        Success = false,
                        Root = x,
                        FunctionValue = fx,
                        Iterations = iteration + 1
                    };
                }

                // ニュートン法の更新
                x = x - fx / derivative;
            }

            // 最大反復回数に達した
            return new NewtonResult
            {
                Success = false,
                Root = x,
                FunctionValue = function(x),
                Iterations = maxIterations
            };
        }

        /// <summary>
        /// ニュートン法による1変数方程式の求解（導関数指定版）
        /// </summary>
        /// <param name="initialGuess">初期推定値</param>
        /// <param name="function">方程式 f(x) = 0 の左辺</param>
        /// <param name="derivative">導関数 f'(x)</param>
        /// <param name="tolerance">収束判定の許容誤差</param>
        /// <param name="maxIterations">最大反復回数</param>
        /// <returns>求解結果</returns>
        public static NewtonResult Newton(
            double initialGuess,
            Func<double, double> function,
            Func<double, double> derivative,
            double tolerance = 1e-6,
            int maxIterations = 100)
        {
            if (function == null)
                throw new ArgumentNullException(nameof(function));
            if (derivative == null)
                throw new ArgumentNullException(nameof(derivative));
            if (tolerance <= 0)
                throw new ArgumentException("許容誤差は正の値である必要があります", nameof(tolerance));

            double x = initialGuess;

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                double fx = function(x);

                // 収束判定
                if (Math.Abs(fx) < tolerance)
                {
                    return new NewtonResult
                    {
                        Success = true,
                        Root = x,
                        FunctionValue = fx,
                        Iterations = iteration + 1
                    };
                }

                double dfx = derivative(x);

                // 導関数が0に近い場合は失敗
                if (Math.Abs(dfx) < 1e-12)
                {
                    return new NewtonResult
                    {
                        Success = false,
                        Root = x,
                        FunctionValue = fx,
                        Iterations = iteration + 1
                    };
                }

                // ニュートン法の更新
                x = x - fx / dfx;
            }

            // 最大反復回数に達した
            return new NewtonResult
            {
                Success = false,
                Root = x,
                FunctionValue = function(x),
                Iterations = maxIterations
            };
        }
    }

    /// <summary>
    /// 最適化結果を格納するクラス
    /// </summary>
    public class OptimizationResult
    {
        /// <summary>最適化が成功したか</summary>
        public bool Success { get; set; }

        /// <summary>最適解</summary>
        public Vector<double> OptimalPoint { get; set; } = Vector<double>.Build.Dense(0);

        /// <summary>最終的な残差のノルム</summary>
        public double ResidualNorm { get; set; }

        /// <summary>反復回数</summary>
        public int Iterations { get; set; }
    }

    /// <summary>
    /// ニュートン法の求解結果を格納するクラス
    /// </summary>
    public class NewtonResult
    {
        /// <summary>求解が成功したか</summary>
        public bool Success { get; set; }

        /// <summary>求めた解（根）</summary>
        public double Root { get; set; }

        /// <summary>解における関数値 f(x)</summary>
        public double FunctionValue { get; set; }

        /// <summary>反復回数</summary>
        public int Iterations { get; set; }
    }
}
