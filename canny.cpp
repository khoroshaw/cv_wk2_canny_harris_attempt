#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<vector>
#include<queue>

int main()
{
    /* cv::Mat的索引和二维数组的索引在同一个坐标系下 */
    cv::Mat lena_img = cv::imread("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_std.png");
    cv::Mat gray_img;
    cv::cvtColor(lena_img, gray_img, cv::COLOR_BGR2GRAY);
    if(gray_img.empty())
    {
        std::cout << "Cannot load image!!!" << std::endl;
        return -1;
    }

    int lena_height = gray_img.rows; // 竖边长度
    int lena_width = gray_img.cols; // 横边长度
    int lena_channels = gray_img.channels(); // 读成灰度值，单通道，1

    int wind_size;
    std::cout << "Define window size with one ODD int: ";
    std::cin >> wind_size;
    int half_wind_size = wind_size / 2;
    double sigma;
    std::cout << "Please enter the sigma value: ";
    std::cin >> sigma;
    std::cout << "Calculating Gaussian weights... " << std::endl;
    std::vector<double> GF_Weights(wind_size);
    double denom_no_PI = 2 * sigma * sigma;
    double sum1 = 0;
    for(int i = 0; i < wind_size; i++) 
    {
        double x2 = (i - half_wind_size) * (i - half_wind_size);
        double log_num = - x2 / (denom_no_PI);
        double num = exp(log_num);
        GF_Weights[i] = num;
        sum1 += GF_Weights[i];
    }   
    for(int i = 0; i < wind_size; i++)
    {
        GF_Weights[i] /= sum1;
    }

    // 二维高斯滤波
    cv::Mat TempMap(wind_size, lena_width, CV_64F, cv::Scalar(0));
    cv::Mat GaussMap(lena_height, lena_width, CV_64F, cv::Scalar(0));
    for(int i = 0; i < lena_height; i++)
    {
        int idx_swin = i % wind_size; // 这么写是没有问题的，unlike ChatGPT suggestions
        uchar * GImgPtr = gray_img.ptr<uchar>(i);
        double * TMapPtr = TempMap.ptr<double>(idx_swin);
        // 行高斯计算，建临时表
        for(int j = 0; j < lena_width; j++)
        {
            TMapPtr[j] = 0; // 清除上一次的临时表数据
            for(int k = -half_wind_size; k <= half_wind_size; k++)
            {
                int idx_calc_row = std::max(0, std::min(lena_width - 1, j + k));
                TMapPtr[j] += GF_Weights[k + half_wind_size] * GImgPtr[idx_calc_row];
            }
        }
        // 列高斯计算，使用临时表
        if(i >= half_wind_size) // 需要未来的half window才可以开始计算
        {
            int idx_calc_col = i - half_wind_size; // 永远在计算half window之前的高斯
            double * GMapPtr = GaussMap.ptr<double>(idx_calc_col);
            for(int j = -half_wind_size; j <= half_wind_size; j++) // 换个思路，这次把列加权计算放外环
            {
                // idx_calc_col为本次列计算的行，idx_real为核计算涉及的行
                int idx_real = std::min(lena_height - 1, std::max(0, idx_calc_col + j)); 
                int idx_wind_col = idx_real % wind_size;
                double * TMapPtr = TempMap.ptr<double>(idx_wind_col); // 如此可以多次重复地用一个TMapPtr
                for(int k = 0; k < lena_width; k++)
                {
                    GMapPtr[k] += GF_Weights[half_wind_size + j] * TMapPtr[k];
                }
            }
        }
    }
    // 补最后的half window的高斯计算，该用的临时表其实已经建立好了
    for(int i = 0; i < half_wind_size; i++)
    {
        // 列高斯计算，使用临时表
        int idx_calc_col = lena_height - half_wind_size + i; // 待计算的行
        double * GMapPtr = GaussMap.ptr<double>(idx_calc_col);
        for(int j = -half_wind_size; j <= half_wind_size; j++)
        {
            int idx_real = std::min(lena_height - 1, std::max(0, idx_calc_col + j));
            int idx_wind_col = idx_real % wind_size; // 临时表对应的行
            double * TMapPtr = TempMap.ptr<double>(idx_wind_col);
            for(int k = 0; k < lena_width; k++)
            {
                GMapPtr[k] += GF_Weights[half_wind_size + j] * TMapPtr[k];
            }
        }
    }

    // cv::imshow("Gray Lena GFed", GaussMap);
    // cv::waitKey(0);
    // cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_gfed.jpg", GaussMap);
    // cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_gray.jpg", gray_img);

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

    return 0;
}