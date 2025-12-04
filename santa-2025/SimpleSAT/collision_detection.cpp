#include <bits/stdc++.h>

//annotated by GPT5
// Theory (English):
// - SAT (Separating Axis Theorem) for convex polygons: test all face normals as candidate axes.
//   If projections of two polygons on any axis are disjoint, a separating axis exists → no overlap.
// - Concave polygons: ear clipping triangulation to split into convex triangles, then test pairwise.
// - OOP design: abstract Shape base + Polygon hierarchy (Convex/Concave) + virtual dispatch;
//   templates allow generic numeric types (float/double).
// - This is pedagogical: clear structure and comments for easy extension to Kaggle packing prototypes.
// 中文说明：
// - 使用 SAT（分离轴定理）实现凸多边形碰撞检测：通过所有边的法向量作为候选分离轴，比较两图形在该轴上的投影区间是否分离。
// - 对凹多边形使用剪耳法做简单三角化，转为多个凸三角形后与对方形状逐一检测。
// - 面向对象设计（OOP）：抽象形状基类 + 多边形派生类（凸/凹）+ 虚函数多态；并使用模板泛型以支持不同数值类型（float/double）。
// - 代码意在教学清晰：每个关键步骤均以注释解释原理与目的，便于扩展到 Kaggle 原型。

namespace geom {

template <typename T>
struct Vec2 {
	T x{}, y{};
	Vec2() = default;
	Vec2(T x_, T y_) : x(x_), y(y_) {}
	// 向量加法
	Vec2<T> operator+(const Vec2<T>& o) const { return {x + o.x, y + o.y}; }
	// 向量减法
	Vec2<T> operator-(const Vec2<T>& o) const { return {x - o.x, y - o.y}; }
	// 数乘（缩放）
	Vec2<T> operator*(T s) const { return {x * s, y * s}; }
};

template <typename T>
inline T dot(const Vec2<T>& a, const Vec2<T>& b)
{ 
	return a.x * b.x + a.y * b.y; 
}

template <typename T>
inline T cross(const Vec2<T>& a, const Vec2<T>& b) 
{ 
	return a.x * b.y - a.y * b.x;
}

template <typename T>
inline Vec2<T> perp(const Vec2<T>& v) 
{
	 return Vec2<T>{-v.y, v.x}; 
}
// English: perp returns a 2D perpendicular vector, used as SAT axis.
// 中文：perp 返回二维向量的一个法向（垂直）向量，用于构造分离轴。

template <typename T>
inline Vec2<T> rotate(const Vec2<T>& v, T deg) {
	T rad = deg * (T)M_PI / (T)180;
	T c = std::cos(rad), s = std::sin(rad);
	return {v.x * c - v.y * s, v.x * s + v.y * c};
}
// English: rotation by degrees, returns a new rotated vector (non-mutating).
// 中文：旋转函数以度数为单位，返回旋转后的新向量，不修改原输入。

// Base class for shapes
template <typename T>
class Shape {
public:
	virtual ~Shape() = default;
	// 平移：所有顶点整体移动
	virtual void translate(const Vec2<T>& t) = 0;
	// 围绕指定原点旋转（单位：度）
	virtual void rotateAround(T deg, const Vec2<T>& origin) = 0;
	// 多态：与另一形状是否重叠
	virtual bool overlaps(const Shape<T>& other) const = 0;
};
// Design Notes (English):
// - Shape offers a common geometry interface enabling polymorphic collision checks.
// - Virtual overlaps enables runtime dispatch: the appropriate algorithm is chosen per type.
// 设计要点（中文）：
// - Shape 提供统一的几何接口，便于不同形状之间进行多态的碰撞检测。
// - overlaps 作为虚函数，使得“谁来决定如何检测”可以在运行时根据对象类型分派。

// Forward declaration
template <typename T>
class ConvexPolygon;

// Polygon base (common utilities)
template <typename T>
class PolygonBase : public Shape<T> {
protected:
	std::vector<Vec2<T>> pts; // 顶点按逆时针（counter-clockwise）
public:
	PolygonBase() = default;
	explicit PolygonBase(std::vector<Vec2<T>> vertices) : pts(std::move(vertices)) {}
	const std::vector<Vec2<T>>& vertices() const { return pts; }

	void translate(const Vec2<T>& t) override {
		// 逐点加上位移向量
		for (auto& p : pts) p = p + t;
	}

	void rotateAround(T deg, const Vec2<T>& origin) override {
		// 先平移到以 origin 为中心的坐标系，旋转后再平移回去
		for (auto& p:pts)
		{
			auto rel=p-origin;
			rel = rotate(rel, deg);
		//relative 相对的
			p = origin+ rel;
		}
	}

	// Project polygon on axis and test overlap
	static std::pair<T, T> projectOnAxis(const std::vector<Vec2<T>>& v, const Vec2<T>& axis) 
	{
//std::vector<Vec2<T>>& v,一个存二维向量的数组
		// 将所有顶点投影到给定轴上，取最小/最大投影值
		T minP = dot(v[0], axis), maxP = minP;
		for (size_t i = 1; i < v.size(); ++i) {
			T d = dot(v[i], axis);
			minP = std::min(minP, d);
			maxP = std::max(maxP, d);
		}
		return {minP, maxP};
	}
};
// Design Notes (English):
// - PolygonBase encapsulates common operations (translate, rotate-around, projection) to reduce duplication.
// - Projections power SAT: disjoint intervals imply a separating axis → no overlap.
// 设计要点（中文）：
// - PolygonBase 封装多边形的共通操作（平移、绕点旋转、投影计算），减少重复代码。
// - 投影用于 SAT：两个区间不相交即可判定“可分离轴存在”，从而不重叠。

// Convex polygon with SAT collision
template <typename T>
class ConvexPolygon : public PolygonBase<T> {
public:
	using PolygonBase<T>::PolygonBase;

	static bool isConvex(const std::vector<Vec2<T>>& v) {
		if (v.size() < 3) return false;
		T sign = 0;
		for (size_t i = 0; i < v.size(); ++i) {
			auto a = v[i];
			auto b = v[(i + 1) % v.size()];
			auto c = v[(i + 2) % v.size()];
			auto ab = b - a;
			auto bc = c - b;
			T z = cross(ab, bc);
			if (z != 0) {
				if (sign == 0) sign = (z > 0 ? 1 : -1);
				else if ((z > 0 ? 1 : -1) != sign) return false;
			}
		}
		return true;
	}
// English: isConvex crudely checks if all turns share the same sign.
// 中文：isConvex 粗略判断顶点序列是否凸（所有转角同号），可用于前置校验与调试。

	bool overlaps(const Shape<T>& other) const override {
		const auto* oConvex = dynamic_cast<const ConvexPolygon<T>*>(&other);
		if (!oConvex) {
			// 回退：未识别为凸多边形时，交由对方类型处理（多态）
			return other.overlaps(*this);
		}
		const auto& A = this->vertices();
		const auto& B = oConvex->vertices();
		// SAT: test all face normals from A and B
		auto testAxes = [&](const std::vector<Vec2<T>>& v) {
			for (size_t i = 0; i < v.size(); ++i) {
				// 使用边的法向量作为分离轴；轴无需单位化，比较投影区间即可
				auto e = v[(i + 1) % v.size()] - v[i];
				auto axis = perp(e);
				auto [aMin, aMax] = PolygonBase<T>::projectOnAxis(A, axis);
				auto [bMin, bMax] = PolygonBase<T>::projectOnAxis(B, axis);
				// 如果区间不相交，则存在分离轴 -> 不重叠
				if (aMax < bMin || bMax < aMin) return false;
			}
			return true;
		};
		return testAxes(A) && testAxes(B);
	}
};
// Key Points (English):
// - SAT tests face normals of both polygons as axes; any disjoint projection ⇒ no overlap.
// - If all axes overlap, no separating axis exists ⇒ overlap.
// 要点（中文）：
// - SAT 需要测试双方所有边的法向量作为分离轴；若任何轴上投影区间互不相交，则两者不重叠。
// - 若所有轴都相交，则没有分离轴，判定重叠。

// Concave polygon: triangulate (ear clipping) then treat as union of convex parts (triangles)
template <typename T>
class ConcavePolygon : public PolygonBase<T> {
private:
	mutable std::vector<ConvexPolygon<T>> convexParts; // lazily built

	static bool isClockwise(const std::vector<Vec2<T>>& v) {
		T area2 = 0; // 近似有符号面积，判断顶点顺序方向
		for (size_t i = 0; i < v.size(); ++i) {
			const auto& a = v[i];
			const auto& b = v[(i + 1) % v.size()];
			area2 += (b.x - a.x) * (b.y + a.y);
		}
		return area2 > 0; // 正值近似表示顺时针（clockwise）
	}

	static bool pointInTriangle(const Vec2<T>& p, const Vec2<T>& a, const Vec2<T>& b, const Vec2<T>& c) {
		// 重心坐标/向量法：判断点 p 是否在三角形 abc 内
		auto v0 = c - a, v1 = b - a, v2 = p - a;
		T dot00 = dot(v0, v0);
		T dot01 = dot(v0, v1);
		T dot02 = dot(v0, v2);
		T dot11 = dot(v1, v1);
		T dot12 = dot(v1, v2);
		T invDenom = (dot00 * dot11 - dot01 * dot01);
		if (invDenom == 0) return false;
		invDenom = 1 / invDenom;
		T u = (dot11 * dot02 - dot01 * dot12) * invDenom;
		T v = (dot00 * dot12 - dot01 * dot02) * invDenom;
		return (u >= 0) && (v >= 0) && (u + v <= 1);
	}

	// Simple ear clipping triangulation for simple polygon (CCW preferred)
	static std::vector<std::array<Vec2<T>, 3>> triangulate(std::vector<Vec2<T>> v) {
		if (v.size() < 3) return {};
		// 保证顶点为逆时针（CCW），便于凸耳判断
		if (isClockwise(v)) std::reverse(v.begin(), v.end());
		std::vector<int> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);
		std::vector<std::array<Vec2<T>, 3>> tris;
		auto isConvexCorner = [&](int i0, int i1, int i2) {
			// 判断 (i0,i1,i2) 是否为凸角（交叉乘积为正表示 CCW）
			auto a = v[i0], b = v[i1], c = v[i2];
			return cross(b - a, c - b) > 0; // CCW
		};
		while (idx.size() >= 3) {
			bool clipped = false;
			for (size_t k = 0; k < idx.size(); ++k) {
				int i0 = idx[(k + idx.size() - 1) % idx.size()];
				int i1 = idx[k];
				int i2 = idx[(k + 1) % idx.size()];
				if (!isConvexCorner(i0, i1, i2)) continue;
				auto a = v[i0], b = v[i1], c = v[i2];
				bool anyInside = false;
				for (size_t m = 0; m < idx.size(); ++m) {
					int im = idx[m];
					if (im == i0 || im == i1 || im == i2) continue;
					if (pointInTriangle(v[im], a, b, c)) { anyInside = true; break; }
				}
				if (anyInside) continue;
				tris.push_back({a, b, c});
				idx.erase(idx.begin() + (long)k);
				clipped = true;
				break;
			}
			if (!clipped) {
				// 兜底：避免死循环，若只剩三个点，直接作为三角形
				if (idx.size() == 3) {
					tris.push_back({v[idx[0]], v[idx[1]], v[idx[2]]});
					idx.clear();
				} else {
					break;
				}
			}
		}
		return tris;
	}

	void ensureConvexParts() const {
		// 惰性构建：首次需要时才进行三角化
		if (!convexParts.empty()) return;
		auto v = this->vertices();
		auto tris = triangulate(v);
		for (auto& tri : tris) {
			std::vector<Vec2<T>> t = {tri[0], tri[1], tri[2]};
			convexParts.emplace_back(t);
		}
	}
// English: ensureConvexParts lazily triangulates only when needed to avoid extra work.
// 中文：ensureConvexParts 采用惰性策略，只有在需要检测时才进行三角化，避免不必要开销。

public:
	using PolygonBase<T>::PolygonBase;

	bool overlaps(const Shape<T>& other) const override {
		ensureConvexParts(); // 将凹多边形拆解为多个三角形（凸部件）
		// Polymorphic double-dispatch simplified: test against convex parts
		const auto* oConvex = dynamic_cast<const ConvexPolygon<T>*>(&other);
		if (oConvex) {
			for (const auto& part : convexParts) {
				if (part.overlaps(*oConvex)) return true;
			}
			return false;
		}
		const auto* oConcave = dynamic_cast<const ConcavePolygon<T>*>(&other);
		if (oConcave) {
			oConcave->ensureConvexParts(); // 双方拆解后，两两部件检测
			for (const auto& a : convexParts) {
				for (const auto& b : oConcave->convexParts) {
					if (a.overlaps(b)) return true;
				}
			}
			return false;
		}
		// Unknown shape type: no overlap by default
		return false;
	}
};
// Key Points (English):
// - Concave polygons are decomposed into convex triangles; perform SAT pairwise.
// - Sufficient for prototypes; replace with stronger decomposition if needed.
// 要点（中文）：
// - 凹多边形通过三角化分解为多个凸三角形，再与对方形状逐一做 SAT 检测。
// - 这种简单近似对大多竞赛原型足够；若需更稳定的分解可替换为更复杂算法（如 Hertel-Mehlhorn）。

// Utility builders
template <typename T>
ConvexPolygon<T> makeRegularPolygon(int sides, T radius, const Vec2<T>& center = {0, 0}, T rotationDeg = 0) {
	std::vector<Vec2<T>> v;
	v.reserve(sides);
	for (int i = 0; i < sides; ++i) {
		T ang = rotationDeg + (T)i * (T)360 / (T)sides;
		// 将半径向量旋转到对应角度，再平移到中心位置
		auto p = Vec2<T>{radius, 0};
		p = rotate(p, ang);
		v.emplace_back(center + p);
	}
	return ConvexPolygon<T>(std::move(v));
}

} // namespace geom

// Demonstration / basic test
int main() {
	using T = double;
	using namespace geom;

	// 构造一个旋转 45° 的正方形和位于 x=3 的正三角形
	auto square = makeRegularPolygon<T>(4, 1.0, {0, 0}, 45.0);
	auto triangle = makeRegularPolygon<T>(3, 1.0, {3, 0}, 0.0);

	bool initialOverlap = square.overlaps(triangle);
	std::cout << "Initial overlap: " << (initialOverlap ? "true" : "false") << "\n";

	// Move triangle closer
	ConvexPolygon<T> triMoved = triangle;
	triMoved.translate({-2.2, 0}); // 向左移动三角形靠近方形
	std::cout << "After translate overlap: " << (square.overlaps(triMoved) ? "true" : "false") << "\n";

	// Concave polygon: a simple arrow shape
	std::vector<Vec2<T>> arrow = {
		{0, 0}, {2, 0}, {2, 1}, {3, 0.0}, {2, -1}, {2, -0}, {0, 0}
	};
	// Remove duplicate last point for a proper polygon
	arrow.pop_back(); // 去掉重复终点，保证简单多边形
	ConcavePolygon<T> concaveArrow(arrow);

	ConvexPolygon<T> box = makeRegularPolygon<T>(4, 0.5, {1.8, 0.0}, 0.0); // 小方块
	std::cout << "Concave vs convex overlap: " << (concaveArrow.overlaps(box) ? "true" : "false") << "\n";

	return 0;
}

