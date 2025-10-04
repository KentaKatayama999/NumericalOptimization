# NumericalOptimization

数値最適化のための再利用可能なユーティリティライブラリ

## 概要

NumericalOptimizationは、非線形最適化問題を解くための汎用的な数値計算ライブラリです。
Levenberg-Marquardt法を中心とした勾配ベースの最適化アルゴリズムを提供します。

## 主な機能

- **ヤコビアン行列の数値微分計算** - ベクトル値関数のヤコビアン行列を自動計算
- **勾配ベクトル計算** - スカラー関数の勾配を数値微分で計算
- **Levenberg-Marquardt法** - 非線形最小二乗法による最適化
- **Vector3との相互変換** - System.Numerics.Vector3とMathNet.Numericsベクトルの変換ユーティリティ

## インストール

### NuGet パッケージ（GitHub Packages）

```bash
dotnet add package NumericalOptimization
```

### プロジェクト参照

```xml
<ItemGroup>
  <ProjectReference Include="path/to/NumericalOptimization/NumericalOptimization.csproj" />
</ItemGroup>
```

## 使用例

### Levenberg-Marquardt法による最適化

```csharp
using NumericalOptimization;
using MathNet.Numerics.LinearAlgebra;

// 初期点を設定
var initialPoint = Vector<double>.Build.Dense(3);
initialPoint[0] = 1.0;
initialPoint[1] = 2.0;
initialPoint[2] = 3.0;

// 残差関数を定義 (例: 3つの方程式)
Func<Vector<double>, Vector<double>> residualFunc = (x) =>
{
    var residual = Vector<double>.Build.Dense(3);
    residual[0] = x[0] * x[0] - 4.0;  // x^2 = 4
    residual[1] = x[1] - 5.0;          // y = 5
    residual[2] = x[2] * x[2] - 9.0;  // z^2 = 9
    return residual;
};

// 最適化を実行
var result = NumericalOptimizer.LevenbergMarquardt(
    initialPoint,
    residualFunc,
    tolerance: 1e-6,
    maxIterations: 100
);

if (result.Success)
{
    Console.WriteLine($"最適解: [{result.OptimalPoint[0]}, {result.OptimalPoint[1]}, {result.OptimalPoint[2]}]");
    Console.WriteLine($"反復回数: {result.Iterations}");
    Console.WriteLine($"残差ノルム: {result.ResidualNorm}");
}
```

### ヤコビアン行列の計算

```csharp
// ベクトル値関数 f: R^2 → R^3
Func<Vector<double>, Vector<double>> function = (x) =>
{
    var result = Vector<double>.Build.Dense(3);
    result[0] = x[0] * x[0] + x[1];
    result[1] = x[0] - x[1] * x[1];
    result[2] = x[0] * x[1];
    return result;
};

var point = Vector<double>.Build.Dense(new[] { 1.0, 2.0 });
var jacobian = NumericalOptimizer.ComputeJacobian(point, function);

// jacobian は 3×2 行列
Console.WriteLine($"ヤコビアン行列:\n{jacobian}");
```

## API リファレンス

### NumericalOptimizer クラス

#### メソッド

- `ComputeJacobian(point, function, stepSize)` - ヤコビアン行列を計算
- `ComputeGradient(point, function, stepSize)` - 勾配ベクトルを計算
- `LevenbergMarquardtStep(currentPoint, residualFunction, lambda, stepSize)` - L-M法の1ステップ実行
- `LevenbergMarquardt(initialPoint, residualFunction, tolerance, maxIterations, initialLambda, lambdaFactor)` - L-M法の完全実行
- `ToMathNetVector(Vector3)` - System.Numerics.Vector3からMathNet.Numericsベクトルに変換
- `ToMathNetVector2D(Vector3)` - Vector3から2次元MathNetベクトルに変換（Z座標を無視）
- `ToVector3(Vector<double>)` - MathNet.NumericsベクトルからVector3に変換

### OptimizationResult クラス

最適化結果を格納するクラス

#### プロパティ

- `Success` (bool) - 最適化が成功したか
- `OptimalPoint` (Vector<double>) - 最適解
- `ResidualNorm` (double) - 最終的な残差のノルム
- `Iterations` (int) - 反復回数

## 依存関係

- **.NET 8.0 / 9.0**
- **MathNet.Numerics 5.0.0** - 線形代数計算
- **System.Numerics** - ベクトル演算

## ライセンス

MIT License

## 関連プロジェクト

- [Geometry](https://github.com/otamo/Geometry) - NURBS曲線・曲面処理ライブラリ（本ライブラリを利用）
