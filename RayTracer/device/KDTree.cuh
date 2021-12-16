#ifndef KDTree_cuh
#define KDTree_cuh

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "IAccelerationStructure.cuh"
#include "../KDTreeNode.h"
#include "../Debug.h"

#include <queue>

namespace rt::device {

class KDTree : public IAccelerationStructure {
public:
    class KDTreeDevice : public IDevice {
    public:
        struct Node {
            size_t m_raytracableCount{ 0 };
            Mesh* m_raytracables{ nullptr };
            AABB m_volume;
        };

        __device__ KDTreeDevice(size_t nodeCount, KDTree::KDTreeDevice::Node* nodes)
            : m_nodeCount(nodeCount)
            , m_nodes(nodes) {}

        __device__ Mesh::RayTraceResult FindClosestHit(const Ray& ray, float minTime, float maxTime, curandState* randState, Mesh::RayTraceRecord& record) const override {
            Mesh::RayTraceResult result = Mesh::RayTraceResult::Missed;
            if(m_nodeCount == 0 || !m_nodes[0].m_volume.Hit(ray, minTime, maxTime)) {
                return result;
            }

            int i = 0;
            int leaf = 0;
            while(i < m_nodeCount) {
                if(m_nodes[i].m_volume.Hit(ray, minTime, maxTime)) {
                    for(int j = 0; j < m_nodes[i].m_raytracableCount; j++) {
                        if(const auto currResult = m_nodes[i].m_raytracables[j].RayTrace(ray, minTime, maxTime, randState, record); currResult != Mesh::RayTraceResult::Missed) {
                            result = currResult;
                            maxTime = record.m_time;
                        }
                    }
                }

                if(i == m_nodeCount - 1) {
                    return result;
                }

                if(i < m_nodeCount / 2) {
                    // Left node
                    i = 2 * i + 1;
                } else {
                    // Right node
                    int k = 1;
                    while(true) {
                        i = (i - 1) / 2;
                        int p = k * 2;
                        if(leaf % p == k - 1) break;
                        k = p;
                    }
                    i = 2 * i + 2;
                    leaf++;
                }
            }

            return result;
        }

    private:
        size_t m_nodeCount;
        KDTree::KDTreeDevice::Node* m_nodes;
    };

    explicit KDTree(unsigned int maxDepth)
        : m_maxDepth(maxDepth) {}

    void PartitionSpace(const MeshList& raytracables) override;

    DevicePtr ToDevice() const {
        return d_KDTree;
    }

private:
    void TreeToArray(KDTreeNode& currentNode, KDTreeDevice::Node* d_array) {
        std::queue<KDTreeNode*> queue;
        queue.push(&currentNode);

        size_t index = 0;
        while(!queue.empty()) {
            KDTreeNode* node = queue.front();

            KDTreeDevice::Node d_node;
            d_node.m_volume = node->m_volume;
            d_node.m_raytracableCount = node->m_raytracables.size();
            if(!node->m_raytracables.empty()) {
                CHECK_CUDA( cudaMalloc((void**)&d_node.m_raytracables, sizeof(Mesh) * node->m_raytracables.size()) );
                for(int i = 0; i < node->m_raytracables.size(); i++) {
                    Mesh mesh = node->m_raytracables[i];
                    CHECK_CUDA( cudaMalloc((void**)&mesh.m_triangles, sizeof(Triangle) * mesh.m_triangleCount) );
                    CHECK_CUDA( cudaMemcpy(mesh.m_triangles, node->m_raytracables[i].m_triangles, sizeof(Triangle) * mesh.m_triangleCount, cudaMemcpyHostToDevice) );

                    CHECK_CUDA( cudaMemcpy(&d_node.m_raytracables[i], &mesh, sizeof(Mesh), cudaMemcpyHostToDevice) );

                    mesh.m_triangles = nullptr;
                }
            } else {
                d_node.m_raytracables = nullptr;
            }
            CHECK_CUDA(cudaMemcpy(&d_array[index], &d_node, sizeof(KDTreeDevice::Node), cudaMemcpyHostToDevice) );
            index++;

            queue.pop();
            if(node->m_left) {
                queue.push(node->m_left.get());
            }

            if(node->m_right) {
                queue.push(node->m_right.get());
            }
        }
    }

    unsigned int m_maxDepth;
    DevicePtr d_KDTree{nullptr};
};

namespace {

__global__ void CreateKDTreeDeviceObject(IAccelerationStructure::DevicePtr kdtreePtr, size_t nodeCount, KDTree::KDTreeDevice::Node* nodes) {
    (*kdtreePtr) = new KDTree::KDTreeDevice(nodeCount, nodes);
}

__global__ void DeleteKDTreeDeviceObject(IAccelerationStructure::DevicePtr kdtreePtr) {
    delete (*kdtreePtr);
}

}

void KDTree::PartitionSpace(const MeshList& raytracables) {
    LOG_INFO("Partition space\n");
    KDTreeNode root(raytracables, m_maxDepth);
    const unsigned int arraySize = static_cast<unsigned int>(std::pow(2U, root.m_depth)) - 1U;

    KDTreeDevice::Node* d_meshTree;
    CHECK_CUDA( cudaMalloc(&d_meshTree, sizeof(KDTreeDevice::Node) * arraySize) );
    TreeToArray(root, d_meshTree);

    CHECK_CUDA( cudaMalloc((void**)&d_KDTree, sizeof(IAccelerationStructure::IDevice)) );
    CreateKDTreeDeviceObject<<<1, 1 >>>(d_KDTree, arraySize, d_meshTree);
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );
    LOG_INFO("Space partitioned\n");
}

}


#endif
