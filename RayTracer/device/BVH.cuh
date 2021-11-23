#ifndef BVH_cuh
#define BHV_cuh

#pragma warning(push, 0)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <curand_kernel.h>
#pragma warning(pop)

#include "IAccelerationStructure.cuh"
#include "../BVHNode.h"
#include "../Debug.h"

#include <queue>

namespace rt::device {

class BVH : public IAccelerationStructure {
public:
    class BVHDevice : public IDevice {
    public:
        struct Node {
            Mesh* m_raytracable;
            AABB m_volume;
        };

        __device__ BVHDevice(size_t nodeCount, Node* nodes)
            : m_nodeCount(nodeCount)
            , m_nodes(nodes) {}

        __device__ Mesh::RayTraceResult FindClosestHit(const Ray& ray, float minTime, float maxTime, curandState* randState, Mesh::RayTraceRecord& record) const override {
            return Mesh::RayTraceResult::Missed;
        }

    private:
        size_t m_nodeCount;
        Node* m_nodes;
    };

    void PartitionSpace(const MeshList& raytracables) override;

    DevicePtr ToDevice() const {
        return d_BVH;
    }

private:
    void TreeToArray(BVHNode& currentNode, BVHDevice::Node* d_array) {
        std::queue<BVHNode*> queue;
        queue.push(&currentNode);

        size_t index = 0;
        while(!queue.empty()) {
            BVHNode* node = queue.front();

            BVHDevice::Node d_node;
            d_node.m_volume = node->m_volume;
            if(node->m_raytracable) {
                Mesh mesh(*node->m_raytracable.get());
                CHECK_CUDA(cudaMalloc((void**)&mesh.m_triangles, sizeof(Triangle) * mesh.m_triangleCount));
                CHECK_CUDA(cudaMemcpy(mesh.m_triangles, node->m_raytracable->m_triangles, sizeof(Triangle) * mesh.m_triangleCount, cudaMemcpyHostToDevice));

                CHECK_CUDA(cudaMalloc(&d_node.m_raytracable, sizeof(Mesh)));
                CHECK_CUDA(cudaMemcpy(d_node.m_raytracable, &mesh, sizeof(Mesh), cudaMemcpyHostToDevice));

                mesh.m_triangles = nullptr;
            } else {
                d_node.m_raytracable = nullptr;
            }
            CHECK_CUDA(cudaMemcpy(&d_array[index], &d_node, sizeof(BVHDevice::Node), cudaMemcpyHostToDevice));
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

    DevicePtr d_BVH{nullptr};
};

namespace {

__global__ void CreateBVHDeviceObject(IAccelerationStructure::DevicePtr bvhPtr, size_t nodeCount, BVH::BVHDevice::Node* nodes) {
    (*bvhPtr) = new BVH::BVHDevice(nodeCount, nodes);
}

__global__ void DeleteBVHDeviceObject(IAccelerationStructure::DevicePtr bvhPtr) {
    delete (*bvhPtr);
}

}

void BVH::PartitionSpace(const MeshList& raytracables) {
    BVHNode root(raytracables);
    const unsigned int arraySize = std::pow(2U, root.m_depth) - 1;

    BVHDevice::Node* d_meshTree;
    CHECK_CUDA( cudaMalloc(&d_meshTree, sizeof(BVHDevice::Node) * arraySize) );
    TreeToArray(root, d_meshTree);

    CHECK_CUDA( cudaMalloc((void**)&d_BVH, sizeof(IAccelerationStructure::IDevice)) );
    CreateBVHDeviceObject<<<1, 1>>>(d_BVH, arraySize, d_meshTree);
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );
}

}

#endif 
