#pragma once

/*
 * Copyright (c) 2024, Chun-Shih Chang <austin20463@gmail.com>
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

#include <modmesh/toggle/RadixTree.hpp>
#include <modmesh/serialization/SerializableItem.hpp>

namespace modmesh
{

class SerializableCallerProfile : SerializableItem
{

public:

    SerializableCallerProfile() = default;
    SerializableCallerProfile(SerializableCallerProfile const &) = default;
    SerializableCallerProfile(SerializableCallerProfile &&) = default;
    SerializableCallerProfile & operator=(SerializableCallerProfile const &) = default;
    SerializableCallerProfile & operator=(SerializableCallerProfile &&) = default;
    ~SerializableCallerProfile() = default;
    SerializableCallerProfile(const CallerProfile & profile)
        : m_caller_name(profile.caller_name)
        , m_total_time(profile.total_time)
        , m_call_count(profile.call_count)
        , m_is_running(profile.is_running)
    {
    }

    MM_DECL_SERIALIZABLE(
        register_member("caller_name", m_caller_name);
        register_member("total_time", static_cast<double>(m_total_time.count() / 1e9));
        register_member("call_count", m_call_count);
        register_member("is_running", m_is_running);)

private:

    std::string m_caller_name;
    std::chrono::nanoseconds m_total_time;
    int m_call_count;
    bool m_is_running;
};

class SerializableRadixTreeNode : SerializableItem
{

public:

    using child_list_type = std::vector<SerializableRadixTreeNode>;
    using key_type = typename RadixTree<CallerProfile>::key_type;

    SerializableRadixTreeNode() = default;
    SerializableRadixTreeNode(SerializableRadixTreeNode const &) = default;
    SerializableRadixTreeNode(SerializableRadixTreeNode &&) = default;
    SerializableRadixTreeNode & operator=(SerializableRadixTreeNode const &) = default;
    SerializableRadixTreeNode & operator=(SerializableRadixTreeNode &&) = default;
    ~SerializableRadixTreeNode() = default;
    SerializableRadixTreeNode(const RadixTreeNode<CallerProfile> * node)
        : m_key(node->key())
        , m_name(node->name())
        , m_data(node->data())
    {
        for (const auto & child : node->children())
        {
            m_children.push_back(SerializableRadixTreeNode(child.get()));
        }
    }

    MM_DECL_SERIALIZABLE(
        register_member("key", m_key);
        register_member("name", m_name);
        register_member("data", m_data);
        register_member("children", m_children);)

private:

    key_type m_key;
    std::string m_name;
    SerializableCallerProfile m_data;
    child_list_type m_children;
};

class SerializableRadixTree : SerializableItem
{

public:

    using key_type = typename RadixTree<CallerProfile>::key_type;
    SerializableRadixTree() = default;
    SerializableRadixTree(SerializableRadixTree const &) = default;
    SerializableRadixTree(SerializableRadixTree &&) = default;
    SerializableRadixTree & operator=(SerializableRadixTree const &) = default;
    SerializableRadixTree & operator=(SerializableRadixTree &&) = default;
    ~SerializableRadixTree() = default;
    SerializableRadixTree(const RadixTree<CallerProfile> & radix_tree)
        : m_root(radix_tree.get_root())
        , m_id_map(radix_tree.get_id_map(RadixTree<CallerProfile>::CallProfilerPK()))
        , m_unique_id(radix_tree.get_unique_node())
    {
    }

    MM_DECL_SERIALIZABLE(
        register_member("radix_tree", m_root);
        register_member("id_map", m_id_map);
        register_member("unique_id", m_unique_id);)

private:

    SerializableRadixTreeNode m_root;
    std::unordered_map<std::string, key_type> m_id_map;
    key_type m_unique_id;
};

/// Utility to serialize and deserialize CallProfiler.
class CallProfilerSerializer
{

public:

    // It returns the json format of the CallProfiler.
    static std::string serialize(const CallProfiler & profiler)
    {
        if (profiler.radix_tree().get_root()->empty_children())
        {
            return "{}";
        }

        SerializableRadixTree serializable_radix_tree(profiler.radix_tree());
        return serializable_radix_tree.to_json();
    }

}; /* end struct CallProfilerSerializer */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
