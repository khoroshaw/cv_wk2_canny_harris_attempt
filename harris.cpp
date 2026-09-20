#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

double calcGaussianFilter(double *weights, int wind_size, double sigma)
{
    std::cout << "Calculating Gaussian weights... " << std::endl;

    double PI = 3.1415926;
    double denom_no_PI = 2 * sigma * sigma;
    double sum1 = 0;
    for(int i = 0; i < wind_size; i++) // 竖边索引
    {
        double x2 = pow(i - wind_size / 2, 2);
        for(int j = 0; j < wind_size; j++) // 横边索引
        {
            double y2 = pow(j - wind_size / 2, 2);
            double log_num = - (x2 + y2) / (denom_no_PI);
            double num = exp(log_num);
            weights[i * wind_size + j] = num / (denom_no_PI * PI);
            sum1 += weights[i * wind_size + j];
        }
    }   
    return sum1;
}

int main()
{
    /* Hyper Parameters*/
    double K = 0.05;
    double Threshold = 1000;

    /* cv::Mat的索引和二维数组的索引在同一个坐标系下 */
    cv::Mat lena_img = cv::imread("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_std.tif");
    cv::Mat gray_img;
    cv::cvtColor(lena_img, gray_img, cv::COLOR_BGR2GRAY);
    if(gray_img.empty())
    {
        std::cout << "Cannot load image!!!" << std::endl;
        return -1;
    }
    cv::imshow("Lena", lena_img);
    cv::waitKey(0);
    cv::imshow("Gray Lena", gray_img);
    cv::waitKey(0);

    int lena_height = gray_img.rows; // 竖边长度
    int lena_width = gray_img.cols; // 横边长度
    int lena_channels = gray_img.channels(); // 读成灰度值，单通道，1

    double *Ix = (double*)calloc(lena_height * lena_width, sizeof(double)); // 横边求导
    double *Iy = (double*)calloc(lena_height * lena_width, sizeof(double)); // 竖边求导

    for(int i = 1; i < lena_height - 1; i++) // 竖边索引
    {
        for(int j = 1; j < lena_width - 1; j++) // 横边索引
        {
            Ix[i * lena_width + j] = ((int)gray_img.at<uchar>(i, j + 1) - (int)gray_img.at<uchar>(i, j - 1)) / 2.0;
            Iy[i * lena_width + j] = ((int)gray_img.at<uchar>(i + 1, j) - (int)gray_img.at<uchar>(i - 1, j)) / 2.0;
        }
    }

    double *kernel_weights;
    int wind_size;
    std::cout << "Define window size with one ODD int: ";
    std::cin >> wind_size;
    kernel_weights = (double*)malloc(wind_size * wind_size * sizeof(double));

    char act_gf;
    double weights_sum = 0;
    std::cout << "Activate Gauss Filter?[Y/N]" << std::endl;
    std::cin >> act_gf;
    if(act_gf == 'Y' || act_gf == 'y')
    {
        double sigma;
        std::cout << "Please enter the sigma value: ";
        std::cin >> sigma;
        weights_sum = calcGaussianFilter(kernel_weights, wind_size, sigma);
    }   
    else
    {
        for(int i = 0; i < wind_size * wind_size; i++)
        {
            kernel_weights[i] = 1.0;
            weights_sum = wind_size * wind_size;
        }
    }

    std::cout << "The kernel weights are: " << std::endl;
    for(int i = 0; i < wind_size; i++)
    {
        for(int j = 0; j < wind_size; j++)
        {
            std::cout << kernel_weights[i * wind_size + j] << ", ";
        }
        std::cout << std::endl;
    }

    int half_wind_size = wind_size / 2;
    double M11 = 0;
    double M12 = 0;
    double M22 = 0;

    // double R_sum = 0;
    double R_max = 0;
    double R_min = 0;
    double *RMap = (double *)calloc(lena_height * lena_width, sizeof(double)); 

    for(int x = half_wind_size; x < lena_height - half_wind_size; x++) // 竖边索引
    {
        for(int y = half_wind_size; y < lena_width - half_wind_size; y++) // 横边索引
        {
            /*  (x, y)表征当前的判断点， 
                而(i, j)是窗口里遍历的点，用于计算M */
            M11 = 0;
            M12 = 0;
            M22 = 0;
            for(int i = -half_wind_size; i <= half_wind_size; i++) // 竖边索引
            {
                for(int j = -half_wind_size; j <= half_wind_size; j++) // 横边索引
                {
                    int idx_w = (i + half_wind_size) * wind_size + j + half_wind_size;
                    int idx_I = (x + i) * lena_width + y + j;
                    M11 += kernel_weights[idx_w] * Ix[idx_I] * Ix[idx_I]; // / weights_sum;
                    M12 += kernel_weights[idx_w] * Ix[idx_I] * Iy[idx_I]; // / weights_sum;
                    M22 += kernel_weights[idx_w] * Iy[idx_I] * Iy[idx_I]; // / weights_sum;
                }
            }
            double R = M11 * M22 - M12 * M12 - K * pow((M11 + M22), 2);
            RMap[x * lena_width + y] = R;
            
            // R_sum += R;
            R_max = (R_max - R < 1e-6) ? R : R_max;
            R_min = (R_min - R > 1e-6) ? R : R_min;
        }
    }

    uchar *RMapUchar = (uchar *)calloc(lena_height * lena_width, sizeof(uchar)); 
    for(int i = 1; i < lena_height - 1; i++) // 竖边索引
    {
        for(int j = 1; j < lena_width - 1; j++) // 横边索引
        {
            RMapUchar[i * lena_width + j] = static_cast<uchar>((RMap[i * lena_width + j] - R_min) / (R_max - R_min) * 255);
        }
    }
    cv::Mat RMap_img(lena_height, lena_width, CV_8UC1, RMapUchar);
    cv::Mat RMap_img_heat(lena_height, lena_width, CV_8UC1);
    cv::applyColorMap(RMap_img, RMap_img_heat, cv::COLORMAP_JET);
    cv::imshow("Lena Heatmap", RMap_img_heat);
    cv::waitKey(0);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_heatmap.jpg", RMap_img_heat);

    Threshold = 0.01 * R_max;
    cv::Mat lena_img2 = lena_img.clone();
    for(int i = 1; i < lena_height - 1; i++) // 竖边索引
    {
        for(int j = 1; j < lena_width - 1; j++) // 横边索引
        {
            if(RMap[i * lena_width + j] - Threshold > 1e-6)
            {
                cv::circle(lena_img2, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2);
            }
        }
    }
    cv::imshow("Lena Edge Points", lena_img2);
    cv::waitKey(0);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_circled.jpg", lena_img2);
    
    Threshold = 0.01 * R_max;
    int nms_padding = 11;
    for(int i = nms_padding; i < lena_height - nms_padding; i++) // 竖边索引
    {
        for(int j = nms_padding; j < lena_width - nms_padding; j++) // 横边索引
        {
            if(RMap[i * lena_width + j] - Threshold < 1e-6)
            {
                continue;
            }
            bool is_max = true;
            for(int dx = -nms_padding; dx <= nms_padding; dx++)
            {
                for(int dy = -nms_padding; dy <= nms_padding; dy++)
                {
                    if(dx == 0 && dy == 0)
                    {
                        continue;
                    }
                    if(RMap[(i + dx) * lena_width + j + dy] - RMap[i * lena_width + j]> 1e-6)
                    {
                        is_max = false;
                        break;
                    }
                }
                if(!is_max)
                {
                    break;
                }
            }
            if(is_max)
            {
                cv::circle(lena_img, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2);
            }
        }
    }
    cv::imshow("Lena Edge Points", lena_img);
    cv::waitKey(0);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_circled_nms.jpg", lena_img);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_gray.jpg", gray_img);

    free(RMap);
    free(RMapUchar);
    free(Ix);
    free(Iy);
    free(kernel_weights);
    return 0;
}