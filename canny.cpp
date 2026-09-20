#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

double calcGaussianFilter(double *weights, int wind_size, double sigma)
{
    std::cout << "Calculating Gaussian weights... " << std::endl;

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
            weights[i * wind_size + j] = num / (denom_no_PI * M_PI);
            sum1 += weights[i * wind_size + j];
        }
    }   
    return sum1;
}

int main()
{
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

    double *kernel_weights;
    int wind_size;
    std::cout << "Define window size with one ODD int: ";
    std::cin >> wind_size;
    kernel_weights = (double*)malloc(wind_size * wind_size * sizeof(double)); 
    double weights_sum = 0;
    double sigma;
    std::cout << "Please enter the sigma value: ";
    std::cin >> sigma;
    weights_sum = calcGaussianFilter(kernel_weights, wind_size, sigma);
    std::cout << "The kernel weights are: " << std::endl;
    for(int i = 0; i < wind_size; i++)
    {
        for(int j = 0; j < wind_size; j++)
        {
            std::cout << kernel_weights[i * wind_size + j] << ", ";
        }
        std::cout << std::endl;
    }

    cv::Mat gray_img_padded;
    int half_wind_size = wind_size / 2;
    cv::copyMakeBorder(gray_img, gray_img_padded, half_wind_size, half_wind_size, half_wind_size, half_wind_size, cv::BORDER_CONSTANT, 0);
    int lena_height_padded = gray_img_padded.rows; // 竖边长度
    int lena_width_padded = gray_img_padded.cols; // 横边长度
    for(int i = half_wind_size; i < lena_height_padded - half_wind_size; i++) // 竖边索引
    {
        for(int j = half_wind_size; j < lena_width_padded - half_wind_size; j++) // 横边索引
        {
            cv::Mat mat_temp = gray_img_padded(cv::Rect(j - half_wind_size, i - half_wind_size, wind_size, wind_size));
            double filtered_val = 0;
            for(int k = 0; k < wind_size * wind_size; k++)
            {
                filtered_val += kernel_weights[k] * static_cast<double>(mat_temp.at<uchar>(k / wind_size, k % wind_size));
            }
            gray_img.at<uchar>(i - half_wind_size, j - half_wind_size) = static_cast<uchar>(filtered_val);
        }
    }
    cv::imshow("Gray Lena Gaussian Filtered", gray_img);
    cv::waitKey(0);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_gray_gaufilt.jpg", gray_img);

    double *Dx = (double*)malloc(lena_height * lena_width * sizeof(double)); // 横边求导
    double *Dy = (double*)malloc(lena_height * lena_width * sizeof(double)); // 竖边求导
    uchar *M = (uchar*)malloc(lena_height * lena_width * sizeof(uchar));
    int *Theta = (int*)malloc(lena_height * lena_width * sizeof(int));
    double m_temp = 0;
    double M_max = 0;
    double M_min = 0;
    double theta_temp = 0;
    double theta_0 = 0;
    double theta_45 = 0;
    double theta_90 = 0;
    double theta_135 = 0;
    double theta_dir = 0;
    cv::Mat gray_img_padded2;
    cv::copyMakeBorder(gray_img, gray_img_padded2, 0, 1, 0, 1, cv::BORDER_CONSTANT, 0);
    for(int i = 0; i < lena_height; i++) // 竖边索引
    {
        for(int j = 0; j < lena_width; j++) // 横边索引
        {
            Dx[i * lena_width + j] = static_cast<double>(gray_img_padded2.at<uchar>(i, j + 1) - gray_img_padded2.at<uchar>(i, j) + gray_img_padded2.at<uchar>(i + 1, j + 1) - gray_img_padded2.at<uchar>(i + 1, j)) / 2.0;
            Dy[i * lena_width + j] = static_cast<double>(gray_img_padded2.at<uchar>(i, j) - gray_img_padded2.at<uchar>(i + 1, j) + gray_img_padded2.at<uchar>(i, j + 1) - gray_img_padded2.at<uchar>(i + 1, j + 1)) / 2.0;
            m_temp = sqrt(pow(Dx[i * lena_width + j], 2) + pow(Dy[i * lena_width + j], 2));
            M_max = fmax(m_temp, M_max);
            M_min = fmin(m_temp, M_min);
            theta_temp = atan2(Dy[i * lena_width + j], Dx[i * lena_width + j]) / M_PI * 180.0;
            theta_0 = (theta_temp > 1e-6) ? fmin(fabs(theta_0), fabs(theta_0 - 180.0)) : fmin(fabs(theta_0), fabs(theta_0 + 180.0));
            theta_45 = (theta_temp > 1e-6) ? fabs(theta_temp - 45.0) : fabs(theta_temp + 135.0);
            theta_90 = (theta_temp > 1e-6) ? fabs(theta_temp - 90.0) : fabs(theta_temp + 90.0);
            theta_135 = (theta_temp > 1e-6) ? fabs(theta_temp - 135.0) : fabs(theta_temp + 45.0);
            theta_dir = fmin(theta_0, fmin(theta_45, fmin(theta_90, theta_135)));
            if(fabs(theta_dir - theta_0) < 1e-6)
                Theta[i * lena_width + j] = 0;
            else if(fabs(theta_dir - theta_45) < 1e-6)
                Theta[i * lena_width + j] = 45;
            else if(fabs(theta_dir - theta_90) < 1e-6)
                Theta[i * lena_width + j] = 90;
            else
                Theta[i * lena_width + j] = 135;
        }
    }
    for(int i = 0; i < lena_height; i++) // 竖边索引
    {
        for(int j = 0; j < lena_width; j++) // 横边索引
        {
            m_temp = sqrt(pow(Dx[i * lena_width + j], 2) + pow(Dy[i * lena_width + j], 2));
            M[i * lena_width + j] = static_cast<uchar>((m_temp - M_min) / (M_max - M_min) * 255.0);
        }
    }           
    cv::Mat lena_grad(lena_height, lena_width, CV_8UC1, M);
    cv::imshow("Gray Lena Gradient Amplitude", lena_grad);
    cv::waitKey(0);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_gray_grad.jpg", lena_grad);

    int idx_for_x = 0; // 竖边索引
    int idx_for_y = 0; // 横边索引
    int idx_back_x = 0; // 竖边索引
    int idx_back_y = 0; // 横边索引
    uchar *NMSMap = (uchar*)calloc(lena_height * lena_width, sizeof(uchar));
    for(int i = 1; i < lena_height - 1; i++) // 竖边索引
    {
        for(int j = 1; j < lena_width - 1; j++) // 横边索引
        {
            if(Theta[i * lena_width + j] == 0)
            {
                idx_for_x = i;
                idx_back_x = i;
                idx_for_y = j + 1;
                idx_back_y = j - 1;
            }
            else if(Theta[i * lena_width + j] == 45)
            {
                idx_for_x = i - 1;
                idx_back_x = i + 1;
                idx_for_y = j + 1;
                idx_back_y = j - 1;
            }
            else if(Theta[i * lena_width + j] == 90)
            {
                idx_for_x = i + 1;
                idx_back_x = i - 1;
                idx_for_y = j;
                idx_back_y = j;
            }
            else
            {
                idx_for_x = i + 1;
                idx_back_x = i - 1;
                idx_for_y = j + 1;
                idx_back_y = j - 1;
            }
            NMSMap[i * lena_width + j] = ((M[i * lena_width + j] > M[idx_back_x * lena_width + idx_back_y]) && (M[i * lena_width + j] > M[idx_for_x * lena_width + idx_for_y])) ? M[i * lena_width + j] : 0;
        }
    }
    cv::Mat lena_nms_grad(lena_height, lena_width, CV_8UC1, NMSMap);
    cv::imshow("Gray Lena Gradient Amplitude NMSed", lena_nms_grad);
    cv::waitKey(0);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_gray_grad_nms.jpg", lena_nms_grad);

    int thrHigh = 10;
    int thrLow = thrHigh * 0.4;
    uchar *NMSMapHigh = (uchar*)calloc(lena_height * lena_width, sizeof(uchar));
    uchar *NMSMapLow = (uchar*)calloc(lena_height * lena_width, sizeof(uchar));
    for(int i = 1; i < lena_height - 1; i++) // 竖边索引
    {
        for(int j = 1; j < lena_width - 1; j++) // 横边索引
        {
            NMSMapHigh[i * lena_width + j] = (NMSMap[i * lena_width + j] > thrHigh) ? 255 : 0;
            NMSMapLow[i * lena_width + j] = (NMSMap[i * lena_width + j] > thrLow) ? 255 : 0;
        }
    }
    cv::Mat lena_nms_grad_high(lena_height, lena_width, CV_8UC1, NMSMapHigh);
    cv::imshow("Gray Lena Gradient Amplitude NMSed High Threshold", lena_nms_grad_high);
    cv::waitKey(0);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_gray_grad_nms_high.jpg", lena_nms_grad_high);
    cv::Mat lena_nms_grad_low(lena_height, lena_width, CV_8UC1, NMSMapLow);
    cv::imshow("Gray Lena Gradient Amplitude NMSed High Threshold", lena_nms_grad_low);
    cv::waitKey(0);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_gray_grad_nms_low.jpg", lena_nms_grad_low);

    for(int i = 1; i < lena_height - 1; i++) // 竖边索引
    {
        for(int j = 1; j < lena_width - 1; j++) // 横边索引
        {
            if(NMSMapHigh[i * lena_width + j] == 0 && NMSMapLow[i * lena_width + j] != 0)
            {
                bool no_edge = true; // true表示周边没点，要舍弃；false表示周边有点，要保留
                for(int dx = -1; dx <= 1; dx++)
                {
                    for(int dy = -1; dy <= 1; dy++)
                    {
                        if(dx == 0 && dy == 0){
                            continue;
                        }
                        if(NMSMapHigh[(i + dx) * lena_width + j + dy] != 0) // 周边有点
                        {
                            no_edge = false;
                            break;
                        }
                    }
                    if(no_edge == false)
                    {
                        break;
                    }
                }
                if(no_edge == false)
                {
                    NMSMapHigh[i * lena_width + j] = 255;
                }
            }
        }
    }
    cv::Mat lena_nms_grad_high_merged(lena_height, lena_width, CV_8UC1, NMSMapHigh);
    cv::imshow("Gray Lena Gradient Amplitude NMSed Threshold Merged", lena_nms_grad_high_merged);
    cv::waitKey(0);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_gray_grad_nms_merged.jpg", lena_nms_grad_high_merged);

    free(Dx);
    free(Dy);
    free(M);
    free(Theta);
    free(NMSMap);
    free(NMSMapHigh);
    free(NMSMapLow);
    free(kernel_weights);
    return 0;
}