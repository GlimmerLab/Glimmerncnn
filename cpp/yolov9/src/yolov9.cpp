#include "layer.h"
#include "net.h"

#include "opencv2/opencv.hpp"

#include <float.h>
#include <stdio.h>
#include <vector>

#define MAX_STRIDE 32

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

float sigmoid(float value) {
    return 1.f / (1.f + std::exp(-value));
}

float softmax(const float *src, float *dst, int length) {
    float alpha = -FLT_MAX;
    for (int c = 0; c < length; c++) {
        float score = src[c];
        if (score > alpha) {
            alpha = score;
        }
    }

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; ++i) {
        dst[i] = std::exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
        dis_sum += (float) i * dst[i];
    }
    return dis_sum;
}

void generate_proposals_det(const ncnn::Mat &feat_blob,
                            std::vector <Object> &objects,
                            int stride,
                            float prob_threshold = 0.25f,
                            int reg_max = 16) {
    float dst[16];

    const int num_c = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;

    const int num_class = num_c - 4 * reg_max;

    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {

            const float *matat = feat_blob.channel(i).row(j);

            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int c = 0; c < num_class; c++) {
                float score = matat[c];
                if (score > class_score) {
                    class_index = c;
                    class_score = score;
                }
            }

            if (class_score >= prob_threshold) {

                float x0 = (float) j + 0.5f - softmax(matat + num_class, dst, reg_max);
                float y0 = (float) i + 0.5f - softmax(matat + num_class + reg_max, dst, reg_max);
                float x1 = (float) j + 0.5f + softmax(matat + num_class + 2 * reg_max, dst, reg_max);
                float y1 = (float) i + 0.5f + softmax(matat + num_class + 3 * reg_max, dst, reg_max);

                x0 *= (float) stride;
                y0 *= (float) stride;
                x1 *= (float) stride;
                y1 *= (float) stride;

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = class_index;
                obj.prob = class_score;
                objects.push_back(obj);
            }
        }
    }
}


float clamp(float val, float mi = 0.f, float ma = 1280.f) {
    return val > mi ? (val < ma ? val : ma) : mi;
}

void scale_coords_det(std::vector <Object> &uncoords,
                      std::vector <Object> &coords,
                      int orin_h,
                      int orin_w,
                      float dh = 0,
                      float dw = 0,
                      float ratio_h = 1.0f,
                      float ratio_w = 1.0f) {
    coords.clear();
    // int dh=top = hpad / 2;
    // int bottom = hpad - hpad / 2;
    // int left = wpad / 2;
    // int dw=right = wpad - wpad / 2;

    for (auto &obj: uncoords) {
        auto &bbox = obj.rect;
        float x0 = (float) bbox.x;
        float y0 = (float) bbox.y;
        float x1 = (float) (bbox.x + bbox.width);
        float y1 = (float) (bbox.y + bbox.height);
        float &score = obj.prob;
        int &label = obj.label;

        x0 = (x0 - dw) / ratio_w;
        y0 = (y0 - dh) / ratio_h;
        x1 = (x1 - dw) / ratio_w;
        y1 = (y1 - dh) / ratio_h;

        x0 = clamp(x0, 1.f, (float) orin_w - 1.f);
        y0 = clamp(y0, 1.f, (float) orin_h - 1.f);
        x1 = clamp(x1, 1.f, (float) orin_w - 1.f);
        y1 = clamp(y1, 1.f, (float) orin_h - 1.f);

        Object new_obj;
        new_obj.rect.x = x0;
        new_obj.rect.y = y0;
        new_obj.rect.width = x1 - x0;
        new_obj.rect.height = y1 - y0;
        new_obj.prob = score;
        new_obj.label = label;
        coords.push_back(new_obj);
    }
}

void non_max_suppression(std::vector <Object> &proposals,
                         std::vector <Object> &results,
                         float conf_thres = 0.25f,
                         float iou_thres = 0.65f) {
    results.clear();
    std::vector <cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    for (auto &pro: proposals) {
        bboxes.push_back(pro.rect);
        scores.push_back(pro.prob);
        labels.push_back(pro.label);
    }

    cv::dnn::NMSBoxes(bboxes, scores, conf_thres, iou_thres, indices);

    for (auto i: indices) {
        results.push_back(proposals[i]);
    }
}

int detect_yolov9(const std::string &modeldir, const cv::Mat &bgr, std::vector <Object> &objects) {

    ncnn::Net yolov9_det;

    yolov9_det.opt.use_vulkan_compute = true;
    // yolov9_det.opt.use_bf16_storage = true;

    // original pretrained model from https://github.com/ultralytics/ultralytics
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    if (yolov9_det.load_param((modeldir + "/yolov9_c.param").c_str()))
        exit(-1);
    if (yolov9_det.load_model((modeldir + "/yolov9_c.bin").c_str()))
        exit(-1);

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
     fprintf(stderr, "%d \n%d \n", w,h);
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / (float) w;
        w = target_size;
        h = (int)(h * scale);
    } else {
        scale = (float) target_size / (float) h;
        h = target_size;
        w = (int)(w * scale);
    }
    fprintf(stderr, "%d \n%d \n", w,h);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    // ultralytics/yolo/data/dataloaders/v5augmentations.py letterbox

    int wpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
     fprintf(stderr, "%d \n", hpad);

    int top = hpad / 2;
    fprintf(stderr, "%d \n", top);
    int bottom = hpad - hpad / 2;
    int left = wpad / 2;
    int right = wpad - wpad / 2;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, top, bottom, left, right, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov9_det.create_extractor();

    ex.input("images", in_pad);

    std::vector <Object> proposals;

    // stride 8
    {
        ncnn::Mat out0;
        ex.extract("output0", out0);

        std::vector <Object> objects8;
        generate_proposals_det(out0, objects8, 8, prob_threshold);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out1;

        ex.extract("1257", out1);

        std::vector <Object> objects16;
        generate_proposals_det(out1, objects16, 16, prob_threshold);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out2;

        ex.extract("1275", out2);

        std::vector <Object> objects32;
        generate_proposals_det(out2, objects32, 32, prob_threshold);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    std::vector <Object> nms_objects;
    non_max_suppression(proposals, nms_objects, prob_threshold, nms_threshold);
    scale_coords_det(nms_objects, objects, img_h, img_w, top, right, scale, scale);

    // for (auto &obj: nms_objects) {
    //     auto &bbox = obj.rect;
    //     float x0 = (float) bbox.x;
    //     float y0 = (float) bbox.y;
    //     float x1 = (float) (bbox.x + bbox.width);
    //     float y1 = (float) (bbox.y + bbox.height);
    //     float &score = obj.prob;
    //     int &label = obj.label;

    //     x0 = (x0 - wpad / 2) / scale;
    //     y0 = (y0 - hpad / 2) / scale;
    //     x1 = (x1 - wpad / 2) / scale;
    //     y1 = (y1 - hpad / 2) / scale;

    //     x0 = clamp(x0, 0.f, (float) img_w - 1.f);
    //     y0 = clamp(y0, 0.f, (float) img_h - 1.f);
    //     x1 = clamp(x1, 0.f, (float) img_w - 1.f);
    //     y1 = clamp(y1, 0.f, (float) img_h - 1.f);

    //     Object new_obj;
    //     new_obj.rect.x = x0;
    //     new_obj.rect.y = y0;
    //     new_obj.rect.width = x1 - x0;
    //     new_obj.rect.height = y1 - y0;
    //     new_obj.prob = score;
    //     new_obj.label = label;
    //     objects.push_back(new_obj);}
    return 0;
}

void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    static const unsigned char colors[19][3] = {
        {54, 67, 244},
        {99, 30, 233},
        {176, 39, 156},
        {183, 58, 103},
        {181, 81, 63},
        {243, 150, 33},
        {244, 169, 3},
        {212, 188, 0},
        {136, 150, 0},
        {80, 175, 76},
        {74, 195, 139},
        {57, 220, 205},
        {59, 235, 255},
        {7, 193, 255},
        {0, 152, 255},
        {34, 87, 255},
        {72, 85, 121},
        {158, 158, 158},
        {139, 125, 96}
    };

    int color_index = 0;

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cc, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char **argv) {
    // if (argc != 3) {
    //     fprintf(stderr, "Usage: %s [imagepath] [modeldir]\n", argv[0]);
    //     return -1;
    // }

    // std::string imagepath = argv[1];
    // std::string modeldir = argv[2];
     std::string imagepath = "../images/test.jpg";
    std::string modeldir = ".";
    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", imagepath.c_str());
        return -1;
    }
    std::vector<Object> objects;

    detect_yolov9(modeldir, m, objects);
    draw_objects(m, objects);
    return 0;
}
