#ifndef IDeviceResource_cuh
#define IDeviceResource_cuh

class IDeviceResource {
public:
    virtual ~IDeviceResource() = default;

    virtual IDeviceResource* ToDevice() = 0;
    virtual void FromDevice(IDeviceResource* host) = 0;
};

#endif
