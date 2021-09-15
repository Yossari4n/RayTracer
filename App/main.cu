#include <RayTracer/PPMTarget.h>

#include <iostream>

int main() {
    rt::PPMTarget target(800, 600);

    for (int j = static_cast<int>(target.Height()) - 1; j >= 0; --j) {
        for (int i = 0; i < target.Width(); ++i) {
            const float r = static_cast<float>(i) / (target.Width() - 1);
            const float g = static_cast<float>(j) / (target.Height() - 1);
            const float b = 0.25;
            target.WriteColor(i, j, rt::Color(r, g, b), 1);
        }
    }

    target.SaveBuffer();

    return 0;
}