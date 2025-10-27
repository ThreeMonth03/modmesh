#pragma once

/*
 * Copyright (c) 2023, Yung-Yu Chen <yyc@solvcon.net>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/base.hpp>
#include <modmesh/buffer/buffer.hpp>
#include <modmesh/universe/bernstein.hpp>

namespace modmesh
{

/**
 * Axis enumeration for 3D space.
 */
enum class Axis
{
    X = 0,
    Y = 1,
    Z = 2
};

/**
 * Point in three-dimensional space.
 *
 * @tparam T floating-point type
 */
template <typename T>
class Point3d
    : public NumberBase<int32_t, T>
{

public:

    static_assert(std::is_arithmetic_v<T>, "T in Point3d<T> must be arithmetic type");

    /**
     * Return the values of the point object.  Do not include the opening and
     * closing parentheses.  Example:
     *   p.value_string(): 0.1234, -2.421, 0
     *
     * @return String of the values separated by comma.
     */
    std::string value_string() const
    {
        return (Formatter() << this->x() << ", " << this->y() << ", " << this->z()).str();
    }

    using value_type = T;

    Point3d(T x, T y)
        : Point3d(x, y, /*z*/ 0.0)
    {
    }

    Point3d(T x, T y, T z)
        : m_coord{x, y, z}
    {
    }

    Point3d() = default;
    Point3d(Point3d const &) = default;
    Point3d(Point3d &&) = default;
    Point3d & operator=(Point3d const &) = default;
    Point3d & operator=(Point3d &&) = default;
    ~Point3d() = default;

    value_type x() const { return m_coord[0]; }
    value_type & x() { return m_coord[0]; }
    void set_x(value_type v) { x() = v; }

    value_type y() const { return m_coord[1]; }
    value_type & y() { return m_coord[1]; }
    void set_y(value_type v) { y() = v; }

    value_type z() const { return m_coord[2]; }
    value_type & z() { return m_coord[2]; }
    void set_z(value_type v) { z() = v; }

    T operator[](size_t i) const { return m_coord[i]; }
    T & operator[](size_t i) { return m_coord[i]; }

    T at(size_t i) const
    {
        check_size(i, 3);
        return m_coord[i];
    }
    T & at(size_t i)
    {
        check_size(i, 3);
        return m_coord[i];
    }

    size_t size() const { return 3; }

    void fill(T v) { m_coord[0] = m_coord[1] = m_coord[2] = v; }

    bool operator==(Point3d const & rhs) const
    {
        return m_coord[0] == rhs.m_coord[0] && m_coord[1] == rhs.m_coord[1] && m_coord[2] == rhs.m_coord[2];
    }

    bool operator!=(Point3d const & rhs) const
    {
        return m_coord[0] != rhs.m_coord[0] || m_coord[1] != rhs.m_coord[1] || m_coord[2] != rhs.m_coord[2];
    }

    Point3d & operator+=(Point3d const & o)
    {
        m_coord[0] += o.m_coord[0];
        m_coord[1] += o.m_coord[1];
        m_coord[2] += o.m_coord[2];
        return *this;
    }

    Point3d & operator-=(Point3d const & o)
    {
        m_coord[0] -= o.m_coord[0];
        m_coord[1] -= o.m_coord[1];
        m_coord[2] -= o.m_coord[2];
        return *this;
    }

    Point3d & operator+=(value_type v)
    {
        m_coord[0] += v;
        m_coord[1] += v;
        m_coord[2] += v;
        return *this;
    }

    Point3d & operator-=(value_type v)
    {
        m_coord[0] -= v;
        m_coord[1] -= v;
        m_coord[2] -= v;
        return *this;
    }

    Point3d & operator*=(value_type v)
    {
        m_coord[0] *= v;
        m_coord[1] *= v;
        m_coord[2] *= v;
        return *this;
    }

    Point3d & operator/=(value_type v)
    {
        m_coord[0] /= v;
        m_coord[1] /= v;
        m_coord[2] /= v;
        return *this;
    }

    value_type calc_length2() const { return m_coord[0] * m_coord[0] + m_coord[1] * m_coord[1] + m_coord[2] * m_coord[2]; }
    value_type calc_length() const { return std::sqrt(calc_length2()); }

    void mirror_x() { m_coord[0] = -m_coord[0]; }
    void mirror_y() { m_coord[1] = -m_coord[1]; }
    void mirror_z() { m_coord[2] = -m_coord[2]; }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("Point3d::mirror: invalid axis"); break;
        }
    }

private:

    void check_size(size_t i, size_t s) const
    {
        if (i >= s)
        {
            throw std::out_of_range(Formatter() << "Point3d: i " << i << " >= size " << s);
        }
    }

    T m_coord[3];

}; /* end class Point3d */

template <typename T>
Point3d<T> operator+(Point3d<T> const & lhs, const Point3d<T> & rhs)
{
    Point3d<T> res = lhs;
    res += rhs;
    return res;
}

template <typename T>
Point3d<T> operator-(Point3d<T> const & lhs, const Point3d<T> & rhs)
{
    Point3d<T> res = lhs;
    res -= rhs;
    return res;
}

template <typename T>
Point3d<T> operator*(Point3d<T> const & lhs, typename Point3d<T>::value_type rhs)
{
    Point3d<T> res = lhs;
    res *= rhs;
    return res;
}

template <typename T>
Point3d<T> operator/(Point3d<T> const & lhs, typename Point3d<T>::value_type rhs)
{
    Point3d<T> res = lhs;
    res /= rhs;
    return res;
}

using Point3dFp32 = Point3d<float>;
using Point3dFp64 = Point3d<double>;

template <typename T>
class PointPad
    : public NumberBase<int32_t, T>
    , public std::enable_shared_from_this<PointPad<T>>
{

private:

    struct ctor_passkey
    {
    };

public:

    using real_type = T;
    using value_type = T;
    using point_type = Point3d<T>;

    template <typename... Args>
    static std::shared_ptr<PointPad<T>> construct(Args &&... args)
    {
        return std::make_shared<PointPad<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    PointPad(uint8_t ndim, ctor_passkey const &)
        : m_ndim(ndim)
    {
        if (ndim > 3)
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "ndim = " << int(ndim) << " > 3");
        }
        else if (ndim < 2)
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "ndim = " << int(ndim) << " < 2");
        }
    }

    PointPad(uint8_t ndim, size_t nelem, ctor_passkey const &)
        : m_ndim(ndim)
        , m_x(nelem)
        , m_y(nelem)
        , m_z()
    {
        if (ndim == 3)
        {
            m_z.expand(nelem);
        }
        else if (ndim > 3)
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "ndim = " << int(ndim) << " > 3");
        }
        else if (ndim < 2)
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "ndim = " << int(ndim) << " < 2");
        }
    }

    // Always clone the input arrays
    PointPad(SimpleArray<T> const & x, SimpleArray<T> const & y, ctor_passkey const &)
        : m_ndim(2)
        , m_x(x)
        , m_y(y)
    {
        if (x.size() != y.size())
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "x.size() " << x.size() << " y.size() " << y.size()
                << " are not the same");
        }
    }

    // Always clone the input arrays
    PointPad(SimpleArray<T> const & x, SimpleArray<T> const & y, SimpleArray<T> const & z, ctor_passkey const &)
        : m_ndim(3)
        , m_x(x)
        , m_y(y)
        , m_z(z)
    {
        if (x.size() != y.size() || x.size() != z.size() || y.size() != z.size())
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "x.size() " << x.size() << " y.size() " << y.size() << " z.size() " << z.size()
                << " are not the same");
        }
    }

    PointPad(SimpleArray<T> & x, SimpleArray<T> & y, bool clone, ctor_passkey const &)
        : m_ndim(2)
        , m_x(x, clone)
        , m_y(y, clone)
    {
        if (x.size() != y.size())
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "x.size() " << x.size() << " y.size() " << y.size()
                << " are not the same");
        }
    }

    PointPad(SimpleArray<T> & x, SimpleArray<T> & y, SimpleArray<T> & z, bool clone, ctor_passkey const &)
        : m_ndim(3)
        , m_x(x, clone)
        , m_y(y, clone)
        , m_z(z, clone)
    {
        if (x.size() != y.size() || x.size() != z.size() || y.size() != z.size())
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "x.size() " << x.size() << " y.size() " << y.size() << " z.size() " << z.size()
                << " are not the same");
        }
    }

    PointPad() = delete;
    PointPad(PointPad const &) = delete;
    PointPad(PointPad &&) = delete;
    PointPad & operator=(PointPad const &) = delete;
    PointPad & operator=(PointPad &&) = delete;

    ~PointPad() = default;

    void append(point_type const & point)
    {
        m_x.push_back(point.x());
        m_y.push_back(point.y());
        if (m_ndim == 3)
        {
            m_z.push_back(point.z());
        }
    }

    void append(T x, T y)
    {
        if (m_ndim != 2)
        {
            throw std::out_of_range(Formatter() << "PointPad::append: ndim must be 2 but is " << int(m_ndim));
        }
        m_x.push_back(x);
        m_y.push_back(y);
    }

    void append(T x, T y, T z)
    {
        if (m_ndim != 3)
        {
            throw std::out_of_range(Formatter() << "PointPad::append: ndim must be 3 but is " << int(m_ndim));
        }
        m_x.push_back(x);
        m_y.push_back(y);
        m_z.push_back(z);
    }

    // Do not implement setter of m_ndim. It should not be changed after construction.
    uint8_t ndim() const { return m_ndim; }

    size_t size() const { return m_x.size(); }

    SimpleArray<T> pack_array() const
    {
        using shape_type = typename SimpleArray<T>::shape_type;
        SimpleArray<T> ret(shape_type{m_x.size(), m_ndim});
        if (m_ndim == 3)
        {
            for (size_t i = 0; i < m_x.size(); ++i)
            {
                ret(i, 0) = m_x[i];
                ret(i, 1) = m_y[i];
                ret(i, 2) = m_z[i];
            }
        }
        else
        {
            for (size_t i = 0; i < m_x.size(); ++i)
            {
                ret(i, 0) = m_x[i];
                ret(i, 1) = m_y[i];
            }
        }
        return ret;
    }

    void expand(size_t length)
    {
        m_x.expand(length);
        m_y.expand(length);
        if (m_ndim == 3)
        {
            m_z.expand(length);
        }
    }

    real_type x_at(size_t i) const { return m_x.at(i); }
    real_type y_at(size_t i) const { return m_y.at(i); }
    real_type z_at(size_t i) const { return m_z.at(i); }
    real_type & x_at(size_t i) { return m_x.at(i); }
    real_type & y_at(size_t i) { return m_y.at(i); }
    real_type & z_at(size_t i) { return m_z.at(i); }

    real_type x(size_t i) const { return m_x[i]; }
    real_type y(size_t i) const { return m_y[i]; }
    real_type z(size_t i) const { return m_z[i]; }
    real_type & x(size_t i) { return m_x[i]; }
    real_type & y(size_t i) { return m_y[i]; }
    real_type & z(size_t i) { return m_z[i]; }

    SimpleArray<value_type> x() { return m_x.as_array(); }
    SimpleArray<value_type> y() { return m_y.as_array(); }
    SimpleArray<value_type> z() { return m_z.as_array(); }

    point_type get_at(size_t i) const
    {
        if (m_ndim == 3)
        {
            return point_type(m_x.at(i), m_y.at(i), m_z.at(i));
        }
        else
        {
            return point_type(m_x.at(i), m_y.at(i), 0.0);
        }
    }
    void set_at(size_t i, point_type const & v)
    {
        m_x.at(i) = v.x();
        m_y.at(i) = v.y();
        if (m_ndim == 3)
        {
            m_z.at(i) = v.z();
        }
    }
    void set_at(size_t i, value_type x, value_type y)
    {
        m_x.at(i) = x;
        m_y.at(i) = y;
    }
    void set_at(size_t i, value_type x, value_type y, value_type z)
    {
        m_x.at(i) = x;
        m_y.at(i) = y;
        if (m_ndim == 3)
        {
            m_z.at(i) = z;
        }
    }

    point_type get(size_t i) const
    {
        if (m_ndim == 3)
        {
            return point_type(m_x[i], m_y[i], m_z[i]);
        }
        else
        {
            return point_type(m_x[i], m_y[i], 0.0);
        }
    }
    void set(size_t i, point_type const & v)
    {
        m_x[i] = v.x();
        m_y[i] = v.y();
        if (m_ndim == 3)
        {
            m_z[i] = v.z();
        }
    }
    void set(size_t i, value_type x, value_type y)
    {
        m_x[i] = x;
        m_y[i] = y;
    }
    void set(size_t i, value_type x, value_type y, value_type z)
    {
        m_x[i] = x;
        m_y[i] = y;
        if (m_ndim == 3)
        {
            m_z[i] = z;
        }
    }

    void mirror_x()
    {
        for (size_t i = 0; i < m_x.size(); ++i)
        {
            m_x[i] = -m_x[i];
        }
    }

    void mirror_y()
    {
        for (size_t i = 0; i < m_y.size(); ++i)
        {
            m_y[i] = -m_y[i];
        }
    }

    void mirror_z()
    {
        if (m_ndim != 3)
        {
            throw std::out_of_range(Formatter() << "PointPad::mirror_z: ndim must be 3 but is " << int(m_ndim));
        }
        for (size_t i = 0; i < m_z.size(); ++i)
        {
            m_z[i] = -m_z[i];
        }
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("PointPad::mirror: invalid axis"); break;
        }
    }

private:

    uint8_t m_ndim;
    SimpleCollector<value_type> m_x;
    SimpleCollector<value_type> m_y;
    // For 2D point pads, m_z should remain unused and empty.
    SimpleCollector<value_type> m_z;

}; /* end class PointPad */

using PointPadFp32 = PointPad<float>;
using PointPadFp64 = PointPad<double>;

namespace detail
{

template <typename T>
struct Segment3dNamed
{
    T x0, x1, y0, y1, z0, z1;
}; /* end struct Segment3dNamed */

template <typename T>
union Segment3dData
{
    T v[6];
    Segment3dNamed<T> f;
}; /* end union Segment3dData */

} /* end namespace detail */

/**
 * Segment in three-dimensional space.
 *
 * @tparam T floating-point type
 */
template <typename T>
class Segment3d
    : public NumberBase<int32_t, T>
{

public:

    static_assert(std::is_arithmetic_v<T>, "T in Segment3d<T> must be arithmetic type");

    using point_type = Point3d<T>;
    using value_type = typename point_type::value_type;

    Segment3d(point_type const & p0, point_type const & p1)
        : m_data{p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), p1.z()}
    {
    }

    Segment3d() = default;
    Segment3d(Segment3d const &) = default;
    Segment3d & operator=(Segment3d const &) = default;
    Segment3d(Segment3d &&) = default;
    Segment3d & operator=(Segment3d &&) = default;
    ~Segment3d() = default;

    point_type p0() const { return point_type(m_data.f.x0, m_data.f.y0, m_data.f.z0); }
    void set_p0(point_type const & p)
    {
        m_data.f.x0 = p.x();
        m_data.f.y0 = p.y();
        m_data.f.z0 = p.z();
    }
    point_type p1() const { return point_type(m_data.f.x1, m_data.f.y1, m_data.f.z1); }
    void set_p1(point_type const & p)
    {
        m_data.f.x1 = p.x();
        m_data.f.y1 = p.y();
        m_data.f.z1 = p.z();
    }

    value_type x0() const { return m_data.f.x0; }
    value_type & x0() { return m_data.f.x0; }
    void set_x0(value_type v) { m_data.f.x0 = v; }

    value_type y0() const { return m_data.f.y0; }
    value_type & y0() { return m_data.f.y0; }
    void set_y0(value_type v) { m_data.f.y0 = v; }

    value_type z0() const { return m_data.f.z0; }
    value_type & z0() { return m_data.f.z0; }
    void set_z0(value_type v) { m_data.f.z0 = v; }

    value_type x1() const { return m_data.f.x1; }
    value_type & x1() { return m_data.f.x1; }
    void set_x1(value_type v) { m_data.f.x1 = v; }

    value_type y1() const { return m_data.f.y1; }
    value_type & y1() { return m_data.f.y1; }
    void set_y1(value_type v) { m_data.f.y1 = v; }

    value_type z1() const { return m_data.f.z1; }
    value_type & z1() { return m_data.f.z1; }
    void set_z1(value_type v) { m_data.f.z1 = v; }

    point_type operator[](size_t i) const { return point_type(m_data.v[i], m_data.v[i + 2], m_data.v[i + 4]); }

    bool operator==(Segment3d const & other) const
    {
        return m_data.v[0] == other.m_data.v[0] &&
               m_data.v[1] == other.m_data.v[1] &&
               m_data.v[2] == other.m_data.v[2] &&
               m_data.v[3] == other.m_data.v[3] &&
               m_data.v[4] == other.m_data.v[4] &&
               m_data.v[5] == other.m_data.v[5];
    }

    bool operator!=(Segment3d const & other) const
    {
        return m_data.v[0] != other.m_data.v[0] ||
               m_data.v[1] != other.m_data.v[1] ||
               m_data.v[2] != other.m_data.v[2] ||
               m_data.v[3] != other.m_data.v[3] ||
               m_data.v[4] != other.m_data.v[4] ||
               m_data.v[5] != other.m_data.v[5];
    }

    point_type at(size_t i) const
    {
        check_size(i, 2);
        return (*this)[i];
    }

    size_t size() const { return 2; }

    void mirror_x()
    {
        m_data.f.x0 = -m_data.f.x0;
        m_data.f.x1 = -m_data.f.x1;
    }

    void mirror_y()
    {
        m_data.f.y0 = -m_data.f.y0;
        m_data.f.y1 = -m_data.f.y1;
    }

    void mirror_z()
    {
        m_data.f.z0 = -m_data.f.z0;
        m_data.f.z1 = -m_data.f.z1;
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("Segment3d::mirror: invalid axis"); break;
        }
    }

private:

    void check_size(size_t i, size_t s) const
    {
        if (i >= s)
        {
            throw std::out_of_range(Formatter() << "Segment3d: i " << i << " >= size " << s);
        }
    }

    detail::Segment3dData<T> m_data;

}; /* end class Segment3d */

using Segment3dFp32 = Segment3d<float>;
using Segment3dFp64 = Segment3d<double>;

template <typename T>
class SegmentPad
    : public NumberBase<int32_t, T>
    , public std::enable_shared_from_this<SegmentPad<T>>
{

private:

    struct ctor_passkey
    {
    };

public:

    using real_type = T;
    using value_type = T;
    using point_type = Point3d<T>;
    using segment_type = Segment3d<T>;
    using point_pad_type = PointPad<T>;

    template <typename... Args>
    static std::shared_ptr<SegmentPad<T>> construct(Args &&... args)
    {
        return std::make_shared<SegmentPad<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    SegmentPad(uint8_t ndim, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim))
        , m_p1(point_pad_type::construct(ndim))
    {
    }

    SegmentPad(uint8_t ndim, size_t nelem, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim, nelem))
        , m_p1(point_pad_type::construct(ndim, nelem))
    {
    }

    SegmentPad(
        SimpleArray<T> const & x0,
        SimpleArray<T> const & y0,
        SimpleArray<T> const & x1,
        SimpleArray<T> const & y1,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0))
        , m_p1(point_pad_type::construct(x1, y1))
    {
        check_constructor_point_size(*m_p0, *m_p1);
    }

    SegmentPad(
        SimpleArray<T> const & x0,
        SimpleArray<T> const & y0,
        SimpleArray<T> const & z0,
        SimpleArray<T> const & x1,
        SimpleArray<T> const & y1,
        SimpleArray<T> const & z1,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, z0))
        , m_p1(point_pad_type::construct(x1, y1, z1))
    {
        check_constructor_point_size(*m_p0, *m_p1);
    }

    SegmentPad(
        SimpleArray<T> & x0,
        SimpleArray<T> & y0,
        SimpleArray<T> & x1,
        SimpleArray<T> & y1,
        bool clone,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, clone))
        , m_p1(point_pad_type::construct(x1, y1, clone))
    {
        check_constructor_point_size(*m_p0, *m_p1);
    }

    SegmentPad(
        SimpleArray<T> & x0,
        SimpleArray<T> & y0,
        SimpleArray<T> & z0,
        SimpleArray<T> & x1,
        SimpleArray<T> & y1,
        SimpleArray<T> & z1,
        bool clone,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, z0, clone))
        , m_p1(point_pad_type::construct(x1, y1, z1, clone))
    {
        check_constructor_point_size(*m_p0, *m_p1);
    }

    std::shared_ptr<SegmentPad<T>> clone()
    {
        if (ndim() == 2)
        {
            return SegmentPad<T>::construct(x0(), y0(), x1(), y1());
        }
        else
        {
            return SegmentPad<T>::construct(x0(), y0(), z0(), x1(), y1(), z1());
        }
    }

    SegmentPad() = delete;
    SegmentPad(SegmentPad const &) = delete;
    SegmentPad(SegmentPad &&) = delete;
    SegmentPad & operator=(SegmentPad const &) = delete;
    SegmentPad & operator=(SegmentPad &&) = delete;

    ~SegmentPad() = default;

    void append(segment_type const & s)
    {
        if (ndim() == 2)
        {
            m_p0->append(s.x0(), s.y0());
            m_p1->append(s.x1(), s.y1());
        }
        else
        {
            m_p0->append(s.x0(), s.y0(), s.z0());
            m_p1->append(s.x1(), s.y1(), s.z1());
        }
    }

    void append(point_type const & p0, point_type const & p1)
    {
        if (ndim() == 2)
        {
            m_p0->append(p0.x(), p0.y());
            m_p1->append(p1.x(), p1.y());
        }
        else
        {
            m_p0->append(p0.x(), p0.y(), p0.z());
            m_p1->append(p1.x(), p1.y(), p1.z());
        }
    }

    void append(T x0, T y0, T x1, T y1)
    {
        m_p0->append(x0, y0);
        m_p1->append(x1, y1);
    }

    void append(T x0, T y0, T z0, T x1, T y1, T z1)
    {
        m_p0->append(x0, y0, z0);
        m_p1->append(x1, y1, z1);
    }

    void extend_with(SegmentPad<T> const & other)
    {
        size_t const nseg = other.size(); // Fix the number since other may be *this
        for (size_t i = 0; i < nseg; ++i)
        {
            append(other.get(i));
        }
    }

    uint8_t ndim() const { return m_p0->ndim(); }

    size_t size() const { return m_p0->size(); }

    SimpleArray<T> pack_array() const
    {
        using shape_type = typename SimpleArray<T>::shape_type;
        SimpleArray<T> ret(shape_type{m_p0->size(), static_cast<size_t>(ndim() * 2)});
        if (ndim() == 3)
        {
            for (size_t i = 0; i < m_p0->size(); ++i)
            {
                ret(i, 0) = m_p0->x(i);
                ret(i, 1) = m_p0->y(i);
                ret(i, 2) = m_p0->z(i);
                ret(i, 3) = m_p1->x(i);
                ret(i, 4) = m_p1->y(i);
                ret(i, 5) = m_p1->z(i);
            }
        }
        else
        {
            for (size_t i = 0; i < m_p0->size(); ++i)
            {
                ret(i, 0) = m_p0->x(i);
                ret(i, 1) = m_p0->y(i);
                ret(i, 2) = m_p1->x(i);
                ret(i, 3) = m_p1->y(i);
            }
        }
        return ret;
    }

    void expand(size_t length)
    {
        m_p0->expand(length);
        m_p1->expand(length);
    }

    real_type x0_at(size_t i) const { return m_p0->x_at(i); }
    real_type y0_at(size_t i) const { return m_p0->y_at(i); }
    real_type z0_at(size_t i) const { return m_p0->z_at(i); }
    real_type x1_at(size_t i) const { return m_p1->x_at(i); }
    real_type y1_at(size_t i) const { return m_p1->y_at(i); }
    real_type z1_at(size_t i) const { return m_p1->z_at(i); }
    real_type & x0_at(size_t i) { return m_p0->x_at(i); }
    real_type & y0_at(size_t i) { return m_p0->y_at(i); }
    real_type & z0_at(size_t i) { return m_p0->z_at(i); }
    real_type & x1_at(size_t i) { return m_p1->x_at(i); }
    real_type & y1_at(size_t i) { return m_p1->y_at(i); }
    real_type & z1_at(size_t i) { return m_p1->z_at(i); }

    real_type x0(size_t i) const { return m_p0->x(i); }
    real_type y0(size_t i) const { return m_p0->y(i); }
    real_type z0(size_t i) const { return m_p0->z(i); }
    real_type x1(size_t i) const { return m_p1->x(i); }
    real_type y1(size_t i) const { return m_p1->y(i); }
    real_type z1(size_t i) const { return m_p1->z(i); }
    real_type & x0(size_t i) { return m_p0->x(i); }
    real_type & y0(size_t i) { return m_p0->y(i); }
    real_type & z0(size_t i) { return m_p0->z(i); }
    real_type & x1(size_t i) { return m_p1->x(i); }
    real_type & y1(size_t i) { return m_p1->y(i); }
    real_type & z1(size_t i) { return m_p1->z(i); }

    point_type p0_at(size_t i) const { return m_p0->get_at(i); }
    point_type p1_at(size_t i) const { return m_p1->get_at(i); }
    void set_p0_at(size_t i, point_type const & p) { m_p0->set_at(i, p); }
    void set_p1_at(size_t i, point_type const & p) { m_p1->set_at(i, p); }

    point_type p0(size_t i) const { return m_p0->get(i); }
    point_type p1(size_t i) const { return m_p1->get(i); }
    void set_p0(size_t i, point_type const & p) { m_p0->set(i, p); }
    void set_p1(size_t i, point_type const & p) { m_p1->set(i, p); }

    SimpleArray<value_type> x0() { return m_p0->x(); }
    SimpleArray<value_type> y0() { return m_p0->y(); }
    SimpleArray<value_type> z0() { return m_p0->z(); }
    SimpleArray<value_type> x1() { return m_p1->x(); }
    SimpleArray<value_type> y1() { return m_p1->y(); }
    SimpleArray<value_type> z1() { return m_p1->z(); }

    // Should not implement the const versions of p0, p1 because the returned
    // shared pointers make the contents mutable.
    std::shared_ptr<point_pad_type> p0() { return m_p0; }
    std::shared_ptr<point_pad_type> p1() { return m_p1; }

    segment_type get_at(size_t i) const
    {
        if (ndim() == 3)
        {
            return segment_type(point_type(x0_at(i), y0_at(i), z0_at(i)), point_type(x1_at(i), y1_at(i), z1_at(i)));
        }
        else
        {
            return segment_type(point_type(x0_at(i), y0_at(i), 0.0), point_type(x1_at(i), y1_at(i), 0.0));
        }
    }
    void set_at(size_t i, segment_type const & s)
    {
        x0_at(i) = s.x0();
        y0_at(i) = s.y0();
        x1_at(i) = s.x1();
        y1_at(i) = s.y1();
        if (ndim() == 3)
        {
            z0_at(i) = s.z0();
            z1_at(i) = s.z1();
        }
    }
    void set_at(size_t i, point_type const & p0, point_type const & p1)
    {
        x0_at(i) = p0.x();
        y0_at(i) = p0.y();
        x1_at(i) = p1.x();
        y1_at(i) = p1.y();
        if (ndim() == 3)
        {
            z0_at(i) = p0.z();
            z1_at(i) = p1.z();
        }
    }
    void set_at(size_t i, value_type x0, value_type y0, value_type x1, value_type y1)
    {
        x0_at(i) = x0;
        y0_at(i) = y0;
        x1_at(i) = x1;
        y1_at(i) = y1;
    }
    void set_at(size_t i, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1)
    {
        x0_at(i) = x0;
        y0_at(i) = y0;
        x1_at(i) = x1;
        y1_at(i) = y1;
        if (ndim() == 3)
        {
            z0_at(i) = z0;
            z1_at(i) = z1;
        }
    }

    segment_type get(size_t i) const
    {
        if (ndim() == 3)
        {
            return segment_type(point_type(x0(i), y0(i), z0(i)), point_type(x1(i), y1(i), z1(i)));
        }
        else
        {
            return segment_type(point_type(x0(i), y0(i), 0.0), point_type(x1(i), y1(i), 0.0));
        }
    }
    void set(size_t i, segment_type const & s)
    {
        x0(i) = s.x0();
        y0(i) = s.y0();
        x1(i) = s.x1();
        y1(i) = s.y1();
        if (ndim() == 3)
        {
            z0(i) = s.z0();
            z1(i) = s.z1();
        }
    }
    void set(size_t i, point_type const & p0, point_type const & p1)
    {
        x0(i) = p0.x();
        y0(i) = p0.y();
        x1(i) = p1.x();
        y1(i) = p1.y();
        if (ndim() == 3)
        {
            z0(i) = p0.z();
            z1(i) = p1.z();
        }
    }
    void set(size_t i, value_type x0, value_type y0, value_type x1, value_type y1)
    {
        x0(i) = x0;
        y0(i) = y0;
        x1(i) = x1;
        y1(i) = y1;
    }
    void set(size_t i, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1)
    {
        x0(i) = x0;
        y0(i) = y0;
        x1(i) = x1;
        y1(i) = y1;
        if (ndim() == 3)
        {
            z0(i) = z0;
            z1(i) = z1;
        }
    }

    void mirror_x()
    {
        size_t const nseg = size();
        for (size_t i = 0; i < nseg; ++i)
        {
            x0(i) = -x0(i);
            x1(i) = -x1(i);
        }
    }

    void mirror_y()
    {
        size_t const nseg = size();
        for (size_t i = 0; i < nseg; ++i)
        {
            y0(i) = -y0(i);
            y1(i) = -y1(i);
        }
    }

    void mirror_z()
    {
        if (ndim() != 3)
        {
            throw std::out_of_range(
                Formatter() << "SegmentPad::mirror_z: cannot mirror Z axis for ndim " << int(ndim()));
        }
        size_t const nseg = size();
        for (size_t i = 0; i < nseg; ++i)
        {
            z0(i) = -z0(i);
            z1(i) = -z1(i);
        }
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("SegmentPad::mirror: invalid axis"); break;
        }
    }

private:

    void check_constructor_point_size(point_pad_type const & p0, point_pad_type const & p1)
    {
        if (m_p0->size() != m_p1->size())
        {
            throw std::invalid_argument(
                Formatter()
                << "SegmentPad::SegmentPad: "
                << "p0.size() " << p0.size() << " p1.size() " << p1.size()
                << " are not the same");
        }
    }

    std::shared_ptr<point_pad_type> m_p0;
    std::shared_ptr<point_pad_type> m_p1;

}; /* end class SegmentPad */

using SegmentPadFp32 = SegmentPad<float>;
using SegmentPadFp64 = SegmentPad<double>;

namespace detail
{

template <typename T>
struct Bezier3dNamed
{
    T x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
}; /* end struct Bezier3dNamed */

template <typename T>
union Bezier3dData
{
    T v[12];
    Bezier3dNamed<T> f;
}; /* end union Segment3dData */

} /* end namespace detail */

/**
 * Bezier curve up to degree 3 in three-dimensional space.
 *
 * @tparam T floating-point type
 */
template <typename T>
class Bezier3d
    : public NumberBase<int32_t, T>
{

public:

    static_assert(std::is_floating_point_v<T>, "T in Bezier3d<T> must be floating-point type");

    using point_type = Point3d<T>;
    using value_type = typename point_type::value_type;

    Bezier3d(point_type const & p0,
             point_type const & p1,
             point_type const & p2,
             point_type const & p3)
        : m_data{p0.x(), p1.x(), p2.x(), p3.x(), p0.y(), p1.y(), p2.y(), p3.y(), p0.z(), p1.z(), p2.z(), p3.z()}
    {
    }

    Bezier3d() = default;
    Bezier3d(Bezier3d const &) = default;
    Bezier3d(Bezier3d &&) = default;
    Bezier3d & operator=(Bezier3d const &) = default;
    Bezier3d & operator=(Bezier3d &&) = default;
    ~Bezier3d() = default;

#define DECL_VALUE_ACCESSOR(C, I)                     \
    value_type C##I() const { return m_data.f.C##I; } \
    value_type & C##I() { return m_data.f.C##I; }
    // clang-format off
    DECL_VALUE_ACCESSOR(x, 0)
    DECL_VALUE_ACCESSOR(x, 1)
    DECL_VALUE_ACCESSOR(x, 2)
    DECL_VALUE_ACCESSOR(x, 3)
    DECL_VALUE_ACCESSOR(y, 0)
    DECL_VALUE_ACCESSOR(y, 1)
    DECL_VALUE_ACCESSOR(y, 2)
    DECL_VALUE_ACCESSOR(y, 3)
    DECL_VALUE_ACCESSOR(z, 0)
    DECL_VALUE_ACCESSOR(z, 1)
    DECL_VALUE_ACCESSOR(z, 2)
    DECL_VALUE_ACCESSOR(z, 3)
    // clang-format on
#undef DECL_VALUE_ACCESSOR

#define DECL_POINT_ACCESSOR(I)                                             \
    point_type p##I() const { return point_type(x##I(), y##I(), z##I()); } \
    void set_p##I(point_type const & p)                                    \
    {                                                                      \
        x##I() = p.x();                                                    \
        y##I() = p.y();                                                    \
        z##I() = p.z();                                                    \
    }
    // clang-format off
    DECL_POINT_ACCESSOR(0)
    DECL_POINT_ACCESSOR(1)
    DECL_POINT_ACCESSOR(2)
    DECL_POINT_ACCESSOR(3)
    // clang-format on
#undef DECL_POINT_ACCESSOR

    std::shared_ptr<SegmentPad<T>> sample(size_t nlocus) const;

    void mirror_x()
    {
        x0() = -x0();
        x1() = -x1();
        x2() = -x2();
        x3() = -x3();
    }

    void mirror_y()
    {
        y0() = -y0();
        y1() = -y1();
        y2() = -y2();
        y3() = -y3();
    }

    void mirror_z()
    {
        z0() = -z0();
        z1() = -z1();
        z2() = -z2();
        z3() = -z3();
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("Bezier3d::mirror: invalid axis"); break;
        }
    }

private:

    void check_size(size_t i, size_t s, char const * msg) const
    {
        if (i >= s)
        {
            throw std::out_of_range(Formatter() << "Bezier3d: (" << msg << ") i " << i << " >= size " << s);
        }
    }

    detail::Bezier3dData<T> m_data;

}; // namespace modmesh

using Bezier3dFp32 = Bezier3d<float>;
using Bezier3dFp64 = Bezier3d<double>;

/**
 * Store curves that are compatible to SVG
 * https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths
 * @tparam T
 */
template <typename T>
class CurvePad
    : public NumberBase<int32_t, T>
    , public std::enable_shared_from_this<CurvePad<T>>
{
private:

    struct ctor_passkey
    {
    };

public:

    using real_type = T;
    using value_type = T;
    using point_type = Point3d<T>;
    using segment_type = Segment3d<T>;
    using bezier_type = Bezier3d<T>;
    using point_pad_type = PointPad<T>;

    template <typename... Args>
    static std::shared_ptr<CurvePad<T>> construct(Args &&... args)
    {
        return std::make_shared<CurvePad<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    CurvePad(uint8_t ndim, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim))
        , m_p1(point_pad_type::construct(ndim))
        , m_p2(point_pad_type::construct(ndim))
        , m_p3(point_pad_type::construct(ndim))
    {
    }

    CurvePad(uint8_t ndim, size_t nelem, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim, nelem))
        , m_p1(point_pad_type::construct(ndim, nelem))
        , m_p2(point_pad_type::construct(ndim, nelem))
        , m_p3(point_pad_type::construct(ndim, nelem))
    {
    }

    CurvePad() = delete;
    CurvePad(CurvePad const &) = delete;
    CurvePad(CurvePad &&) = delete;
    CurvePad & operator=(CurvePad const &) = delete;
    CurvePad & operator=(CurvePad &&) = delete;

    ~CurvePad() = default;

    void append(bezier_type const & c)
    {
        m_p0->append(c.p0());
        m_p1->append(c.p1());
        m_p2->append(c.p2());
        m_p3->append(c.p3());
    }

    void append(point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
    {
        m_p0->append(p0);
        m_p1->append(p1);
        m_p2->append(p2);
        m_p3->append(p3);
    }

    uint8_t ndim() const { return m_p0->ndim(); }

    size_t size() const { return m_p0->size(); }

    SimpleArray<T> pack_array() const
    {
        using shape_type = typename SimpleArray<T>::shape_type;
        SimpleArray<T> ret(shape_type{m_p0->size(), static_cast<size_t>(ndim() * 4)});
        if (ndim() == 3)
        {
            for (size_t i = 0; i < m_p0->size(); ++i)
            {
                ret(i, 0) = m_p0->x(i);
                ret(i, 1) = m_p0->y(i);
                ret(i, 2) = m_p0->z(i);
                ret(i, 3) = m_p1->x(i);
                ret(i, 4) = m_p1->y(i);
                ret(i, 5) = m_p1->z(i);
                ret(i, 6) = m_p2->x(i);
                ret(i, 7) = m_p2->y(i);
                ret(i, 8) = m_p2->z(i);
                ret(i, 9) = m_p3->x(i);
                ret(i, 10) = m_p3->y(i);
                ret(i, 11) = m_p3->z(i);
            }
        }
        else
        {
            for (size_t i = 0; i < m_p0->size(); ++i)
            {
                ret(i, 0) = m_p0->x(i);
                ret(i, 1) = m_p0->y(i);
                ret(i, 2) = m_p1->x(i);
                ret(i, 3) = m_p1->y(i);
                ret(i, 4) = m_p2->x(i);
                ret(i, 5) = m_p2->y(i);
                ret(i, 6) = m_p3->x(i);
                ret(i, 7) = m_p3->y(i);
            }
        }
        return ret;
    }

    void expand(size_t length)
    {
        m_p0->expand(length);
        m_p1->expand(length);
        m_p2->expand(length);
        m_p3->expand(length);
    }

#define DECL_VALUE_ACCESSOR(I, C)                                     \
    real_type C##I##_at(size_t i) const { return m_p##I->C##_at(i); } \
    real_type & C##I##_at(size_t i) { return m_p##I->C##_at(i); }     \
    real_type C##I(size_t i) const { return m_p##I->C(i); }           \
    real_type & C##I(size_t i) { return m_p##I->C(i); }
    // clang-format off
    DECL_VALUE_ACCESSOR(0, x)
    DECL_VALUE_ACCESSOR(0, y)
    DECL_VALUE_ACCESSOR(0, z)
    DECL_VALUE_ACCESSOR(1, x)
    DECL_VALUE_ACCESSOR(1, y)
    DECL_VALUE_ACCESSOR(1, z)
    DECL_VALUE_ACCESSOR(2, x)
    DECL_VALUE_ACCESSOR(2, y)
    DECL_VALUE_ACCESSOR(2, z)
    DECL_VALUE_ACCESSOR(3, x)
    DECL_VALUE_ACCESSOR(3, y)
    DECL_VALUE_ACCESSOR(3, z)
    // clang-format on
#undef DECL_VALUE_ACCESSOR

#define DECL_POINT_ACCESSOR(I)                                                          \
    point_type p##I##_at(size_t i) const { return m_p##I->get_at(i); }                  \
    void set_p##I##_at(size_t i, point_type const & p) { return m_p##I->set_at(i, p); } \
    point_type p##I(size_t i) const { return m_p##I->get(i); }                          \
    void set_p##I(size_t i, point_type const & p) { return m_p##I->set(i, p); }
    // clang-format off
    DECL_POINT_ACCESSOR(0)
    DECL_POINT_ACCESSOR(1)
    DECL_POINT_ACCESSOR(2)
    DECL_POINT_ACCESSOR(3)
    // clang-format on
#undef DECL_POINT_ACCESSOR

    bezier_type get_at(size_t i) const
    {
        return bezier_type(p0_at(i), p1_at(i), p2_at(i), p3_at(i));
    }
    void set_at(size_t i, bezier_type const & c)
    {
        m_p0->set_at(i, c.p0());
        m_p1->set_at(i, c.p1());
        m_p2->set_at(i, c.p2());
        m_p3->set_at(i, c.p3());
    }

    bezier_type get(size_t i) const
    {
        return bezier_type(p0(i), p1(i), p2(i), p3(i));
    }
    void set(size_t i, bezier_type const & c)
    {
        m_p0->set(i, c.p0());
        m_p1->set(i, c.p1());
        m_p2->set(i, c.p2());
        m_p3->set(i, c.p3());
    }

    // TODO: missing many accessors

    SimpleArray<value_type> x0() { return m_p0->x(); }
    SimpleArray<value_type> y0() { return m_p0->y(); }
    SimpleArray<value_type> z0() { return m_p0->z(); }
    SimpleArray<value_type> x1() { return m_p1->x(); }
    SimpleArray<value_type> y1() { return m_p1->y(); }
    SimpleArray<value_type> z1() { return m_p1->z(); }
    SimpleArray<value_type> x2() { return m_p2->x(); }
    SimpleArray<value_type> y2() { return m_p2->y(); }
    SimpleArray<value_type> z2() { return m_p2->z(); }
    SimpleArray<value_type> x3() { return m_p3->x(); }
    SimpleArray<value_type> y3() { return m_p3->y(); }
    SimpleArray<value_type> z3() { return m_p3->z(); }

    // Should not implement the const versions of p0, p1, p2, p3 because the
    // returned shared pointers make the contents mutable.
    std::shared_ptr<point_pad_type> p0() { return m_p0; }
    std::shared_ptr<point_pad_type> p1() { return m_p1; }
    std::shared_ptr<point_pad_type> p2() { return m_p2; }
    std::shared_ptr<point_pad_type> p3() { return m_p3; }

    std::shared_ptr<SegmentPad<T>> sample(value_type length) const;

    void mirror_x()
    {
        m_p0->mirror_x();
        m_p1->mirror_x();
        m_p2->mirror_x();
        m_p3->mirror_x();
    }

    void mirror_y()
    {
        m_p0->mirror_y();
        m_p1->mirror_y();
        m_p2->mirror_y();
        m_p3->mirror_y();
    }

    void mirror_z()
    {
        m_p0->mirror_z();
        m_p1->mirror_z();
        m_p2->mirror_z();
        m_p3->mirror_z();
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("CurvePad::mirror: invalid axis"); break;
        }
    }

private:

    std::shared_ptr<point_pad_type> m_p0;
    std::shared_ptr<point_pad_type> m_p1;
    std::shared_ptr<point_pad_type> m_p2;
    std::shared_ptr<point_pad_type> m_p3;

}; /* end class CurvePad */

using CurvePadFp32 = CurvePad<float>;
using CurvePadFp64 = CurvePad<double>;

template <typename T>
class CubicBezierSampler
{
public:
    using point_type = Point3d<T>;
    using bezier_type = Bezier3d<T>;
    using curve_pad_type = CurvePad<T>;
    using segment_pad_type = SegmentPad<T>;

    using value_type = typename bezier_type::value_type;

    CubicBezierSampler(size_t ndim)
        : m_segments(segment_pad_type::construct(ndim))
    {
    }

    CubicBezierSampler() = delete;
    CubicBezierSampler(CubicBezierSampler const &) = delete;
    CubicBezierSampler(CubicBezierSampler &&) = default;
    CubicBezierSampler & operator=(CubicBezierSampler const &) = delete;
    CubicBezierSampler & operator=(CubicBezierSampler &&) = default;
    ~CubicBezierSampler() = default;

    std::shared_ptr<segment_pad_type> operator()(bezier_type const & curve, size_t nlocus, bool inplace);
    std::shared_ptr<segment_pad_type> operator()(curve_pad_type const & curves, T length);

    void reset() { m_segments.swap(segment_pad_type::construct(/*ndim*/ m_segments.ndim())); }

private:

    static size_t calc_nlocus(Bezier3d<T> const & c, T length);

    static size_t sample_to(bezier_type const & c, segment_pad_type & segment, size_t nlocus);

    std::shared_ptr<segment_pad_type> m_segments;
}; /* end class CubicBezierSampler */

template <typename T>
size_t CubicBezierSampler<T>::calc_nlocus(Bezier3d<T> const & c, T length)
{
    // The verbose code helps debugger stepping, but I did not check the assembly for performance
    point_type const & tp0 = c.p0();
    point_type const & tp3 = c.p3();
    point_type const vec = tp3 - tp0;
    value_type val = vec.calc_length();
    val /= length;
    val = std::floor(val);
    return val < 2 ? 2 : static_cast<size_t>(val);
}

template <typename T>
size_t CubicBezierSampler<T>::sample_to(bezier_type const & c, segment_pad_type & segments, size_t nlocus)
{
    point_type const & tp0 = c.p0();
    point_type const & tp1 = c.p1();
    point_type const & tp2 = c.p2();
    point_type const & tp3 = c.p3();
    point_type lastp = tp0;
    size_t nseg = 0;
    for (size_t j = 1; j < nlocus - 1; ++j)
    {
        value_type t = j;
        t /= nlocus - 1;
        point_type thisp;
        for (size_t idim = 0; idim < 3; ++idim)
        {
            std::vector<T> cvalues{tp0[idim], tp1[idim], tp2[idim], tp3[idim]};
            thisp[idim] = detail::interpolate_bernstein_impl(t, cvalues, cvalues.size() - 1);
        }
        segments.append(lastp, thisp);
        ++nseg;
        lastp = thisp;
    }
    segments.append(lastp, tp3);
    ++nseg;
    return nseg;
}

// Sample for all curves in a curve pad sequentially
template <typename T>
std::shared_ptr<SegmentPad<T>> CubicBezierSampler<T>::operator()(curve_pad_type const & curves, T length)
{
    for (size_t i = 0; i < curves.size(); ++i)
    {
        // Determine number of locus
        bezier_type const & c = curves.get(i);
        size_t const nlocus = calc_nlocus(c, length);
        // No sample and append the base line segment (p0-p3)
        if (nlocus <= 2)
        {
            m_segments->append(c.p0(), c.p3());
        }
        // Sample
        else
        {
            sample_to(c, *m_segments, nlocus);
        }
    }
    return m_segments;
}

template <typename T>
std::shared_ptr<SegmentPad<T>> CurvePad<T>::sample(value_type length) const
{
    return CubicBezierSampler<T>(/*ndim*/ 3)(*this, length);
}

// Sample for all curves in a curve pad sequentially
template <typename T>
std::shared_ptr<SegmentPad<T>> CubicBezierSampler<T>::operator()(bezier_type const & curve, size_t nlocus, bool inplace)
{
    std::shared_ptr<segment_pad_type> segments = inplace ? m_segments : segment_pad_type::construct(m_segments->ndim());
    // No sample and append the base line segment (p0-p3)
    if (nlocus <= 2)
    {
        segments->append(curve.p0(), curve.p3());
    }
    // Sample
    else
    {
        sample_to(curve, *segments, nlocus);
    }
    return segments;
}

template <typename T>
std::shared_ptr<SegmentPad<T>> Bezier3d<T>::sample(size_t nlocus) const
{
    return CubicBezierSampler<T>(/*ndim*/ 3)(*this, nlocus, /*inplace*/ false);
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
