// ===================== sat_bindings.cpp =====================
// English: Pybind11 module exposing C++ SAT geometry to Python.
// 中文：通过 Pybind11 将 C++ 的 SAT 几何功能暴露给 Python。
// ------------------------------------------------------------
#include <pybind11/pybind11.h> // EN: Core pybind11 definitions. CN：pybind11 主要接口。
#include <pybind11/stl.h>      // EN: Enables automatic STL conversions. CN：启用 STL 容器自动转换。
#include "geom.hpp"            // EN: Our geometry header. CN：几何头文件。

namespace py = pybind11; // EN: Alias for convenience. CN：方便书写的别名。
using namespace geom;    // EN: Use geometry namespace. CN：使用几何命名空间。

using Poly = ConvexPolygon<double>; // EN: Type alias for double precision polygon. CN：double 精度凸多边形别名。

PYBIND11_MODULE(satgeom, m) { // EN: Module init function; name must match Extension. CN：模块初始化函数；名称需与扩展一致。
    m.doc() = "SAT geometry bindings (convex polygons)"; // EN: Module docstring. CN：模块文档字符串。

    py::class_<Vec2<double>>(m, "Vec2") // EN: Bind Vec2 class. CN：绑定 Vec2 类。
        .def(py::init<double,double>())   // EN: Constructor with x,y. CN：构造函数（x,y）。
        .def_readwrite("x", &Vec2<double>::x) // EN: Expose field x. CN：暴露成员 x。
        .def_readwrite("y", &Vec2<double>::y); // EN: Expose field y. CN：暴露成员 y。

    py::class_<Poly>(m, "ConvexPolygon") // EN: Bind convex polygon. CN：绑定凸多边形。
        .def(py::init([](std::vector<Vec2<double>> pts){ return Poly(std::move(pts)); })) // EN: Construct from vertex list. CN：用顶点列表构造。
        .def("vertices", &Poly::vertices, py::return_value_policy::reference_internal) // EN: Return internal reference (no copy). CN：返回内部引用（不复制）。
        .def("translate", &Poly::translate)     // EN: Expose translation. CN：暴露平移函数。
        .def("rotateAround", &Poly::rotateAround) // EN: Expose rotation. CN：暴露绕点旋转。
        .def("overlaps", [](const Poly &a, const Poly &b){ return a.overlaps(b); }); // EN: Overlap test wrapper. CN：重叠检测包装。

        m.def("make_regular_polygon", &makeRegularPolygon<double>, // EN: Factory function binding. CN：绑定工厂函数。
            py::arg("sides"), py::arg("radius"), py::arg("center") = Vec2<double>(0,0), py::arg("rotationDeg")=0.0,
            "Build a regular convex polygon."); // EN: Docstring. CN：文档说明。

    m.def("bounding_square_side", &boundingSquareSide<double>, "Axis-aligned bounding square side length."); // EN: Expose bounding helper. CN：暴露包围正方形计算。
    m.def("any_overlap", &anyOverlap<double>, "Return True if any pair overlaps."); // EN: Expose overlap check. CN：暴露任意重叠检测。
}
