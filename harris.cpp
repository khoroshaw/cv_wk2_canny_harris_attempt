#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<vector>

int main()
{
    /* Hyper Parameters*/
    double K = 0.05;
    int nms_padding = 7;
    
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

    // 没必要储存Ix和Iy
    // 本算法用水平和竖直两个高斯核实现二维高斯滤波，以下为滤波时需要计算的值
    cv::Mat Ix2(lena_height, lena_width, CV_64F);
    cv::Mat Iy2(lena_height, lena_width, CV_64F); 
    cv::Mat IxIy(lena_height, lena_width, CV_64F); 

    for(int i = 0; i < lena_height; i++) // 竖边索引
    {
        double * Ix2ptr = Ix2.ptr<double>(i);
        double * Iy2ptr = Iy2.ptr<double>(i);
        double * IxIyptr = IxIy.ptr<double>(i);
        uchar * Gi0ptr = gray_img.ptr<uchar>(((i == 0) ? 0 : (i - 1)));
        uchar * Gi1ptr = gray_img.ptr<uchar>(i);
        uchar * Gi2ptr = gray_img.ptr<uchar>(((i == lena_height - 1) ? (lena_height - 1) : (i + 1)));
        for(int j = 0; j < lena_width; j++) // 横边索引
        {
            double Ixptr_j = (Gi1ptr[((j == lena_width - 1) ? (lena_width - 1) : (j + 1))] - Gi1ptr[((j == 0) ? 0 : (j - 1))]) / 2.0;
            double Iyptr_j = (Gi2ptr[j] - Gi0ptr[j]) / 2.0;
            // 顺手把这些先算好，为了分布高斯滤波
            Ix2ptr[j] = Ixptr_j * Ixptr_j;
            Iy2ptr[j] = Iyptr_j * Iyptr_j;
            IxIyptr[j] = Ixptr_j * Iyptr_j;
        }
    }

    int wind_size;
    double sigma;
    std::cout << "Define window size with one ODD int: ";
    std::cin >> wind_size;    
    std::cout << "Please enter the sigma value: ";
    std::cin >> sigma;
    
    std::cout << "Calculating Gaussian weights... " << std::endl;
    std::vector<double> Kernel_Weights_1d(wind_size); // 一维高斯实现二维卷积，降低时间复杂度
    double denom_no_PI = 2 * sigma * sigma;
    double sum1 = 0;
    for(int i = 0; i < wind_size; i++) 
    {
        double x2 = pow(i - wind_size / 2, 2);
        double log_num = - x2 / denom_no_PI;
        Kernel_Weights_1d[i] = exp(log_num);
        sum1 += Kernel_Weights_1d[i];
    }   
    for(int i = 0; i < wind_size; i++) 
    {
        Kernel_Weights_1d[i] /= sum1;
    }   

    int half_wind_size = wind_size / 2;

    // 计算行权相加  
    cv::Mat M11TempMap(lena_height, lena_width, CV_64F, cv::Scalar(0)); 
    cv::Mat M12TempMap(lena_height, lena_width, CV_64F, cv::Scalar(0)); 
    cv::Mat M22TempMap(lena_height, lena_width, CV_64F, cv::Scalar(0)); 
    
    for(int x = 0; x < lena_height; x++) // 竖边索引
    {
        double * M11ptr = M11TempMap.ptr<double>(x);
        double * M12ptr = M12TempMap.ptr<double>(x);
        double * M22ptr = M22TempMap.ptr<double>(x);
        double * Ix2Eptr = Ix2.ptr<double>(x);
        double * IxIyEptr = IxIy.ptr<double>(x);
        double * Iy2Eptr = Iy2.ptr<double>(x);
        for(int y = 0; y < lena_width; y++) // 横边索引
        {
            /*  (x, y)表征当前gray_img的判断点   */
            for(int i = -half_wind_size; i <= half_wind_size; i++)
            {
                int idx = std::max(0, std::min(y + i, lena_width - 1));
                M11ptr[y] += Kernel_Weights_1d[i + half_wind_size] * Ix2Eptr[idx];
                M12ptr[y] += Kernel_Weights_1d[i + half_wind_size] * IxIyEptr[idx];
                M22ptr[y] += Kernel_Weights_1d[i + half_wind_size] * Iy2Eptr[idx];
            }
        }
    }

    // 计算列权相加
    cv::Mat RMap(lena_height, lena_width, CV_64F, cv::Scalar(0));

    for(int x = 0; x < lena_height; x++) // 竖边索引
    {

        double * Rptr = RMap.ptr<double>(x);
        for(int y = 0; y < lena_width; y++) // 横边索引
        {
            /*  (x, y)表征当前gray_img的判断点   */
            double M11ptr_y = 0;
            double M12ptr_y = 0;
            double M22ptr_y = 0;
            for(int i = -half_wind_size; i <= half_wind_size; i++)
            {
                int idx = std::max(0, std::min(x + i, lena_height - 1));
                double * M11TempMapExtptr = M11TempMap.ptr<double>(idx);
                double * M12TempMapExtptr = M12TempMap.ptr<double>(idx);
                double * M22TempMapExtptr = M22TempMap.ptr<double>(idx);
                M11ptr_y += Kernel_Weights_1d[i + half_wind_size] * M11TempMapExtptr[y];
                M12ptr_y += Kernel_Weights_1d[i + half_wind_size] * M12TempMapExtptr[y];
                M22ptr_y += Kernel_Weights_1d[i + half_wind_size] * M22TempMapExtptr[y];
            }
            Rptr[y] = M11ptr_y * M22ptr_y - M12ptr_y * M12ptr_y - K * (M11ptr_y + M22ptr_y) * (M11ptr_y + M22ptr_y);
        }
    }

    double Rmin;
    double Rmax;
    cv::Mat RMapflatten = RMap.reshape(0, 1);
    cv::minMaxLoc(RMapflatten, &Rmin, &Rmax);
    double Threshold = 0.01 * Rmax;
    cv::Mat lena_img2 = lena_img.clone();
    for(int i = 1; i < lena_height - 1; i++) // 竖边索引
    {
        double * RMptr = RMap.ptr<double>(i);
        for(int j = 1; j < lena_width - 1; j++) // 横边索引
        {
            if(RMptr[j] - Threshold > 1e-6)
            {
                cv::circle(lena_img2, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2);
            }
        }
    }
    cv::imshow("Lena Edge Points", lena_img2);
    cv::waitKey(0);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_circled.jpg", lena_img2);
    
    for(int i = nms_padding; i < lena_height - nms_padding; i++) // 竖边索引
    {
        double * RMptr = RMap.ptr<double>(i);
        for(int j = nms_padding; j < lena_width - nms_padding; j++) // 横边索引
        {
            if(RMptr[j] - Threshold < 1e-6)
            {
                continue;
            }
            bool is_max = true;
            for(int dx = -nms_padding; dx <= nms_padding; dx++)
            {
                double * RMdptr = RMap.ptr<double>(i + dx);
                for(int dy = -nms_padding; dy <= nms_padding; dy++)
                {
                    if(dx == 0 && dy == 0)
                    {
                        continue;
                    }
                    
                    if(RMdptr[j + dy] - RMptr[j] > 1e-6)
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

    return 0;
}