// ===================== geom.hpp =====================
// English: Header-only geometry utilities for convex polygons with SAT collision tests.
// 中文：头文件，提供凸多边形与分离轴定理 (SAT) 碰撞检测的工具（模板+简单函数）。
// -----------------------------------------------------
#pragma once // English: Prevent multiple inclusion. 中文：防止重复包含。
#include <vector>      // English: dynamic array container. 中文：动态数组容器。
#include <cmath>       // English: math functions (cos, sin, etc.). 中文：数学函数。
#include <numeric>     // English: accumulation utilities (not heavily used here). 中文：数值工具。
#include <algorithm>   // English: min/max, move, etc. 中文：常用算法（最值、移动）。

namespace geom { // English: Namespace groups geometry types. 中文：命名空间，将几何类型集中管理。

template <typename T>
struct Vec2 { // English: Simple 2D vector with basic operations. 中文：简单二维向量，含基本运算。
    T x{}, y{}; // English: Components initialized to zero. 中文：分量初始化为0。

    Vec2() = default; // English: Default constructor. 中文：默认构造。

    Vec2(T x_, T y_) : x(x_), y(y_) {} // English: Construct with given x,y. 中文：用给定 x,y 构造。

    Vec2<T> operator+(const Vec2<T>& o) const { return {x + o.x, y + o.y}; } // EN: Vector addition. CN：向量加法。

    Vec2<T> operator-(const Vec2<T>& o) const { return {x - o.x, y - o.y}; } // EN: Vector subtraction. CN：向量减法。

    Vec2<T> operator*(T s) const { return {x * s, y * s}; } // EN: Scalar multiply. CN：数乘缩放。
};

template <typename T>
inline T dot(const Vec2<T>& a, const Vec2<T>& b) { return a.x * b.x + a.y * b.y; } // EN: Dot product (projection measure). CN：点乘，衡量投影与夹角。

template <typename T>
inline T cross(const Vec2<T>& a, const Vec2<T>& b) { return a.x * b.y - a.y * b.x; } // EN: 2D cross (signed area). CN：二维叉乘（符号面积）。

template <typename T>
inline Vec2<T> perp(const Vec2<T>& v) { return Vec2<T>{-v.y, v.x}; } // EN: Perpendicular vector (rotate 90°). CN：垂直向量（旋转90度）。

template <typename T>
inline Vec2<T> rotate(const Vec2<T>& v, T deg) { // EN: Rotate vector by degrees. CN：按角度（度）旋转向量。
    T rad = deg * (T)M_PI / (T)180; // EN: Convert degrees -> radians. CN：角度转弧度。
    T c = std::cos(rad), s = std::sin(rad); // EN: Precompute cos/sin. CN：预计算余弦/正弦。
    return {v.x * c - v.y * s, v.x * s + v.y * c}; // EN: 2D rotation formula. CN：二维旋转公式。
}

template <typename T>
class Shape { // EN: Abstract base for geometric shapes. CN：几何形状抽象基类。
public:
    virtual ~Shape() = default; // EN: Virtual destructor for polymorphic cleanup. CN：虚析构，确保多态正确释放。

    virtual void translate(const Vec2<T>& t) = 0; // EN: Move by vector t. CN：按向量 t 平移。

    virtual void rotateAround(T deg, const Vec2<T>& origin) = 0; // EN: Rotate around point. CN：绕指定点旋转。

    virtual bool overlaps(const Shape<T>& other) const = 0; // EN: Collision test with another shape. CN：与另一形状的碰撞检测。
};

template <typename T>
class ConvexPolygon; // EN: Forward declaration. CN：前向声明。

template <typename T>
class PolygonBase : public Shape<T> { // EN: Common polygon operations. CN：多边形通用操作基类。
protected:
    std::vector<Vec2<T>> pts; // CCW order. 中文：逆时针顶点顺序。
public:
    PolygonBase() = default; // EN: Default construct empty polygon. CN：默认构造空多边形。
    explicit PolygonBase(std::vector<Vec2<T>> vertices) : pts(std::move(vertices)) {} // EN: Take ownership of vertex list. CN：接管顶点列表。
    const std::vector<Vec2<T>>& vertices() const { return pts; } // EN: Access vertices. CN：访问顶点。
    void translate(const Vec2<T>& t) override 
    {
        for (auto &p: pts) 
                p = p + t;  // EN: Add translation to each vertex. CN：逐顶点加平移。
    }
    void rotateAround(T deg, const Vec2<T>& origin) override 
    { // EN: Rotate all vertices around origin. CN：围绕 origin 旋转所有顶点。
        for (auto &p: pts) 
        {
            auto rel = p - origin;
            rel = rotate(rel, deg);
            p = origin + rel;
        } // EN: Shift to origin, rotate, shift back. CN：平移到原点→旋转→移回。
    }
    static std::pair<T,T> projectOnAxis(const std::vector<Vec2<T>>& v, const Vec2<T>& axis) 
    { 
        // EN: Project polygon onto axis for SAT. CN：将多边形投影到轴上用于 SAT。
        T minP = dot(v[0], axis), maxP = minP;  // EN: Initialize min/max projection. CN：初始化投影最小/最大值。

        for (size_t i = 1; i < v.size(); ++i) 
        {
            T d = dot(v[i], axis);
            minP = std::min(minP, d);
            maxP = std::max(maxP, d);
        }

        return {minP, maxP};  // EN: Update extremes. CN：更新投影范围。
    }
};

template <typename T>
class ConvexPolygon : public PolygonBase<T>
{
     // EN: Convex polygon with SAT overlap test. CN：带 SAT 检测的凸多边形。
public:
    using PolygonBase<T>::PolygonBase; 
    // EN: Inherit constructors. CN：继承构造函数。
    bool overlaps(const Shape<T>& other) const override 
    { 
        // EN: Polymorphic collision test. CN：多态碰撞检测。
        auto* oConvex = dynamic_cast<const ConvexPolygon<T>*>(&other);  
        // EN: Try cast to convex polygon. CN：尝试转换为凸多边形。

        if (!oConvex) return other.overlaps(*this);  
        // EN: Defer if different type. CN：不同类型→交由对方处理。

        const auto& A = this->vertices();  
        // EN: This polygon's vertices. CN：当前多边形顶点。
        const auto& B = oConvex->vertices();  
        // EN: Other polygon's vertices. CN：另一多边形顶点。

        auto testAxes = [&](const std::vector<Vec2<T>>& v)
        {  
            // EN: Test axes from edges. CN：测试边法向轴。
            for (size_t i = 0; i < v.size(); ++i) {
                auto e = v[(i + 1) % v.size()] - v[i];
                auto axis = perp(e);

                auto [aMin, aMax] = PolygonBase<T>::projectOnAxis(A, axis);
                auto [bMin, bMax] = PolygonBase<T>::projectOnAxis(B, axis);

                if (aMax < bMin || bMax < aMin) return false; 
                 // EN: Separating axis. CN：分离轴。
            }
            return true;
        };

        return testAxes(A) && testAxes(B);  
    // EN: Must pass axes from both. CN：需通过双方轴测试。
    }
};
 // ==========================================
        // 新手注释 / Beginner Note:
        // 1. auto: 让编译器自动推断类型。这里它代表一个“匿名函数”（Lambda）。
        // 2. [&]: 引用捕获。意思是这个函数可以直接使用外面的变量（如 A 和 B），
        //    而不需要把它们复制进来（省内存、速度快）。
        // ==========================================
template <typename T>
ConvexPolygon<T> makeRegularPolygon(int sides, T radius, const Vec2<T>& center={0,0}, T rotationDeg=0)
{ 
    // EN: Build regular polygon. CN：构造正多边形。
    std::vector<Vec2<T>> v; v.reserve(sides);
     // EN: Reserve memory. CN：预留空间。
    for(int i=0;i<sides;++i){
        T ang = rotationDeg + (T)i * (T)360 / (T)sides; 
        // EN: Angle for vertex i. CN：第 i 个顶点角度。
        auto p=Vec2<T>{radius,0};
        p=rotate(p,ang);
        v.emplace_back(center+p);
    } 
    // EN: Rotate radius vector & translate. CN：旋转半径向量后平移到中心。
    return ConvexPolygon<T>(std::move(v)); 
    // EN: Return constructed polygon. CN：返回构造好的多边形。
}

// Helper: bounding square side (axis-aligned) for set of convex polygons
// English: Compute side length of minimal axis-aligned square covering all polygons.
// 中文：计算包围所有多边形的最小轴对齐正方形的边长。
template <typename T>
T boundingSquareSide(const std::vector<ConvexPolygon<T>>& polys){
    if(polys.empty()) return (T)0; // EN: No polygons ⇒ side 0. CN：空集合→边长 0。
    bool first=true; T minx=0,miny=0,maxx=0,maxy=0; // EN: Track extrema. CN：记录最小/最大坐标。
    for(auto &poly: polys){ // EN: Iterate polygons. CN：遍历多边形。
        for(auto &p: poly.vertices()){ // EN: Iterate vertices. CN：遍历顶点。
            if(first){
                minx=maxx=p.x;
                miny=maxy=p.y;
                first=false;
            } // EN: Initialize with first vertex. CN：用首顶点初始化范围。
            else {
                minx=std::min(minx,p.x);
                miny=std::min(miny,p.y);
                maxx=std::max(maxx,p.x);
                maxy=std::max(maxy,p.y);
            } // EN: Update bounds. CN：更新边界值。
        }
    }
    T w = maxx - minx; 
    T h = maxy - miny; 
    return std::max(w,h); // EN: Square side = max(width,height). CN：正方形边长为宽高较大者。
}

// Check any overlaps
// English: Returns true if any pair of polygons overlaps (O(N^2)).
// 中文：任意一对多边形重叠则返回 true（O(N^2)）。
template <typename T>
bool anyOverlap(const std::vector<ConvexPolygon<T>>& polys){
    for(size_t i=0;i<polys.size();++i) // EN: Outer loop over polygons. CN：外层循环遍历多边形。
        for(size_t j=i+1;j<polys.size();++j) // EN: Inner loop (unique pairs). CN：内层循环（唯一配对）。
            if(polys[i].overlaps(polys[j])) return true; // EN: Early exit on first overlap. CN：发现重叠立即返回。
    return false; // EN: No overlaps found. CN：未发现重叠。
}

} // namespace geom // EN: End geometry namespace. CN：结束几何命名空间。
