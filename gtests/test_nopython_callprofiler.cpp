#include <gtest/gtest.h>
#include <thread>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

#define CALLPROFILER 1
#include <modmesh/toggle/RadixTree.hpp>
namespace modmesh
{

namespace detail
{
class CallProfilerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CallProfiler & profiler = CallProfiler::instance();
        pProfiler = &profiler;
    }

    RadixTree<CallerProfile> & radix_tree()
    {
        return pProfiler->m_radix_tree;
    }

    CallProfiler * pProfiler;
};

constexpr int uniqueTime1 = 19;
constexpr int uniqueTime2 = 35;
constexpr int uniqueTime3 = 7;

void foo3()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime1)
    {
        // use busy loop to get a precise duration
    }
}

void foo2()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime2)
    {
        // use busy loop to get a precise duration
    }
    foo3();
}

void foo1()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    foo2();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime3)
    {
        // use busy loop to get a precise duration
    }
}

TEST_F(CallProfilerTest, test_print_result)
{
    pProfiler->reset();

    foo1();

    std::stringstream ss;
    pProfiler->print_profiling_result(ss);
}

static bool diff_time(std::chrono::nanoseconds raw_nano_time, int time_ms)
{
    constexpr int error = 5; // a reasonable error
    return std::abs(raw_nano_time.count() / 1e6 - time_ms) < error;
}

#ifdef _MSC_VER
auto foo1Name = "void __cdecl modmesh::detail::foo1(void)";
#else
auto foo1Name = "void modmesh::detail::foo1()";
#endif

#ifdef _MSC_VER
auto foo2Name = "void __cdecl modmesh::detail::foo2(void)";
#else
auto foo2Name = "void modmesh::detail::foo2()";
#endif

#ifdef _MSC_VER
auto foo3Name = "void __cdecl modmesh::detail::foo3(void)";
#else
auto foo3Name = "void modmesh::detail::foo3()";
#endif

TEST_F(CallProfilerTest, test_simple_case1)
{
    pProfiler->reset();

    foo1();

    // Example:
    // void modmesh::foo1() - Total Time: 61 ms, Call Count: 1
    //   void modmesh::foo2() - Total Time: 54 ms, Call Count: 1
    //      void modmesh::foo3() - Total Time: 19 ms, Call Count: 1

    int key = 0;

    auto * node1 = radix_tree().get_current_node()->get_child(key++);
    EXPECT_EQ(node1->data().caller_name, foo1Name);
    EXPECT_EQ(node1->data().call_count, 1);
    EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 + uniqueTime2 + uniqueTime3));

    auto * node2 = node1->get_child(key++);
    EXPECT_EQ(node2->data().caller_name, foo2Name);
    EXPECT_EQ(node2->data().call_count, 1);
    EXPECT_TRUE(diff_time(node2->data().total_time, uniqueTime1 + uniqueTime2));

    auto * node3 = node2->get_child(key++);
    EXPECT_EQ(node3->data().caller_name, foo3Name);
    EXPECT_EQ(node3->data().call_count, 1);
    EXPECT_TRUE(diff_time(node3->data().total_time, uniqueTime1));
}

TEST_F(CallProfilerTest, simple_case_2)
{
    pProfiler->reset();

    foo1();
    foo2();
    foo3();
    foo3();

    // Example:
    // void modmesh::foo1 - Total Time: 61 ms, Call Count: 1
    //   void modmesh::foo2 - Total Time: 54 ms, Call Count: 1
    //     void modmesh::foo3 - Total Time: 19 ms, Call Count: 1
    // void modmesh::foo2 - Total Time: 54 ms, Call Count: 1
    //   void modmesh::foo3 - Total Time: 19 ms, Call Count: 1
    // void modmesh::foo3 - Total Time: 38 ms, Call Count: 2

    // for first `foo1()` call
    {
        auto * node1 = radix_tree().get_current_node()->get_child(foo1Name);
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, foo1Name);
        EXPECT_EQ(node1->data().call_count, 1);
        EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 + uniqueTime2 + uniqueTime3));

        auto * node2 = node1->get_child(foo2Name);
        EXPECT_NE(node2, nullptr);
        EXPECT_EQ(node2->data().caller_name, foo2Name);
        EXPECT_EQ(node2->data().call_count, 1);
        EXPECT_TRUE(diff_time(node2->data().total_time, uniqueTime1 + uniqueTime2));

        auto * node3 = node2->get_child(foo3Name);
        EXPECT_NE(node3, nullptr);
        EXPECT_EQ(node3->data().caller_name, foo3Name);
        EXPECT_EQ(node3->data().call_count, 1);
        EXPECT_TRUE(diff_time(node3->data().total_time, uniqueTime1));
    }

    // for  `foo2()` call
    {
        auto * node1 = radix_tree().get_current_node()->get_child(foo2Name); // id = 1, because previously already assigned in the map, FIXME: probably find a better way than hard
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, foo2Name);
        EXPECT_EQ(node1->data().call_count, 1);
        EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 + uniqueTime2));

        auto * node2 = node1->get_child(foo3Name);
        EXPECT_NE(node2, nullptr);
        EXPECT_EQ(node2->data().caller_name, foo3Name);
        EXPECT_EQ(node2->data().call_count, 1);
        EXPECT_TRUE(diff_time(node2->data().total_time, uniqueTime1));
    }

    // for  two `foo3()` call
    {
        auto * node1 = radix_tree().get_current_node()->get_child(foo3Name);
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, foo3Name);
        EXPECT_EQ(node1->data().call_count, 2);
        EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 * 2));
    }
}

TEST_F(CallProfilerTest, cancel)
{
    pProfiler->reset();

    auto test1 = [&]()
    {
        USE_CALLPROFILER_PROFILE_THIS_FUNCTION();

        auto test2 = [&]()
        {
            USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
            pProfiler->cancel();
        };

        test2();
    };
    test1();

    EXPECT_EQ(radix_tree().get_unique_node(), 0);
}

std::string line2 = "";
std::string line_header = static_cast<std::string>("                           Function Name") +
                          "               Call Count" +
                          "           Total Time (s)" +
                          "             Per Call (s)" +
                          "      Cumulative Time (s)" +
                          "             Per Call (s)";

TEST_F(CallProfilerTest, test_statistic)
{
    pProfiler->reset();

    foo1();
    foo2();
    foo3();
    foo3();

    // Example:
    //  7 function calls in 0.153001 seconds
    //
    //                       Function Name               Call Count           Total Time (s)             Per Call (s)      Cumulative Time (s)             Per Call (s)
    //        void modmesh::detail::foo1()                        1                0.0610011                0.0610011               0.00700021               0.00700021
    //        void modmesh::detail::foo2()                        2                 0.108001                0.0540006                0.0700011                0.0350006
    //        void modmesh::detail::foo3()                        4                0.0760001                    0.019                0.0760001                    0.019

    // for first `foo1()` call
    std::stringstream ss;
    pProfiler->print_statistics(ss);

    std::string line;

    // Check the total time and call count
    getline(ss, line);
    int ref_call_count = 7;
    int total_call_count = line[6] - '0';
    EXPECT_EQ(total_call_count, ref_call_count);

    double ref_time = 4 * uniqueTime1 + 2 * uniqueTime2 + uniqueTime3;
    double total_time = std::stod(line.substr(26, 10)) * 1e3;
    EXPECT_TRUE(abs(ref_time - total_time) < 5);

    // The line 2 is empty
    getline(ss, line);
    EXPECT_EQ(line, "");

    // The line 3 is the header
    getline(ss, line);
    EXPECT_EQ(line, line_header);

    // Read the function information
    for (int i = 0; i < 3; ++i)
    {
        // read the words in the line
        getline(ss, line);
        std::stringstream ss_word(line);

        std::string func_type;
        std::string func_name = "";
        int call_count;
        double ttime;
        double per_call_ttime;
        double ctime;
        double per_call_ctime;

        ss_word >> func_type;
        if (func_type == "void")
        {
            ss_word >> func_name;
            func_name = func_type + " " + func_name;
        }
        else
        {
            func_name = func_type;
        }

        ss_word >> call_count;
        ss_word >> ttime;
        ss_word >> per_call_ttime;
        ss_word >> ctime;
        ss_word >> per_call_ctime;

        // The function name should be one of the three functions
        EXPECT_TRUE(func_name == foo1Name || func_name == foo2Name || func_name == foo3Name);

        int ref_call_count;
        double ref_ttime;
        double ref_per_call_ttime;
        double ref_ctime;
        double ref_per_call_ctime;

        // The reference values
        if (func_name == foo1Name)
        {
            ref_call_count = 1;
            ref_ttime = uniqueTime1 + uniqueTime2 + uniqueTime3;
            ref_ctime = uniqueTime3;
        }
        else if (func_name == foo2Name)
        {
            ref_call_count = 2;
            ref_ttime = (uniqueTime1 + uniqueTime2) * 2;
            ref_ctime = uniqueTime2 * 2;
        }
        else if (func_name == foo3Name)
        {
            ref_call_count = 4;
            ref_ttime = uniqueTime1 * 4;
            ref_ctime = uniqueTime1 * 4;
        }

        ref_per_call_ttime = ref_ttime / ref_call_count;
        ref_per_call_ctime = ref_ctime / ref_call_count;

        // Check the values
        EXPECT_EQ(call_count, ref_call_count);
        EXPECT_TRUE(abs(ttime * 1e3 - ref_ttime) < 5);
        EXPECT_TRUE(abs(per_call_ttime * 1e3 - ref_per_call_ttime) < 5);
        EXPECT_TRUE(abs(ctime * 1e3 - ref_ctime) < 5);
        EXPECT_TRUE(abs(per_call_ctime * 1e3 - ref_per_call_ctime) < 5);
    }
}

} // namespace detail
} // namespace modmesh