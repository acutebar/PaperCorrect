#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int r;
  int c;
} COORD;

int disc_create(int radius, COORD *coords) {
  int count = 0;
  for (int r = -radius; r <= radius; r++) {
    for (int c = -radius; c <= radius; c++) {
      if (r*r + c*c <= radius*radius) {
        coords[count].r = r;
        coords[count].c = c;
        count++;
      }
    }
  }
  return count;
}

int disc_mean(unsigned char *img, int rows, int cols, COORD center) {
}


int main() {
    int width, height, channels;
    
    // Load image into 1D array. Force grayscale by setting desired_channels to 1 (0 keeps original).
    unsigned char *img = stbi_load("curvedsurface1.jpeg", &width, &height, &channels, 1);
    
    if (img == NULL) {
        printf("Error loading image.\n");
        return 1;
    }

    // Number of pixels (since forced grayscale, channels=1)
    size_t img_size = width * height;

    // Dummy operation: Invert colors
    for (size_t i = 0; i < img_size; i++) {
        img[i] = 255 - img[i];
    }

    // Output to file
    stbi_write_jpg("output.jpg", width, height, 1, img, 100);

    stbi_image_free(img);
    
    return 0;
}
