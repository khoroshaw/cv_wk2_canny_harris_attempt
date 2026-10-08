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

    // Gemini建议的Lambda表达式，由于两块代码是一样的，因此可以让代码更美观
    // 此处实现的是高斯滤波的列计算
    auto calc_col_gauss = [&](int row)
    {
        int idx_calc_col = row; // 永远在计算half window之前的高斯，待计算的行
        double * GMapPtr = GaussMap.ptr<double>(idx_calc_col);
        for(int j = -half_wind_size; j <= half_wind_size; j++) // 换个思路，这次把列加权计算放外环
        {
            // idx_calc_col为本次列计算的行，idx_real为核计算涉及的行
            int idx_real = std::min(lena_height - 1, std::max(0, idx_calc_col + j)); 
            int idx_wind_col = idx_real % wind_size; // 临时表对应的行
            double * TMapPtr = TempMap.ptr<double>(idx_wind_col); // 如此可以多次重复地用一个TMapPtr
            double weight = GF_Weights[half_wind_size + j];
            for(int k = 0; k < lena_width; k++)
            {
                GMapPtr[k] += weight * TMapPtr[k];
            }
        }
    };

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
            calc_col_gauss(i - half_wind_size);
        }
    }
    // 补最后的half window的高斯计算，该用的临时表其实已经建立好了
    for(int i = 0; i < half_wind_size; i++)
    {
        calc_col_gauss(lena_height - half_wind_size + i);
    }

    // 计算梯度幅值和方向，方向归为4类：0, 45, 90, 135
    // 有意思的是，没有必要计算准确的角度值
    cv::Mat Magnitude(lena_height, lena_width, CV_64F);
    cv::Mat Theta(lena_height, lena_width, CV_8U);
    for(int i = 0; i < lena_height; i++)
    {
        uchar * GImgPtr0 = gray_img.ptr<uchar>(i);
        uchar * GImgPtr1 = gray_img.ptr<uchar>((i == lena_height - 1) ? lena_height - 1 : i + 1);
        double * MagPtr = Magnitude.ptr<double>(i);
        uchar * ThePtr = Theta.ptr<uchar>(i);
        for(int j = 0; j < lena_width; j++)
        {
            int j1 = (j == lena_width - 1) ? lena_width - 1 : j + 1;
            // 右为正方向，两次右减左
            double Dx = (GImgPtr0[j1] - GImgPtr0[j] + GImgPtr1[j1] - GImgPtr1[j]) / 2.0;
            // 下为正方向，两次下减上，这里和老师的ppt有点出入
            // 实际上应该用Sobel算子，不过无所谓
            double Dy = (GImgPtr1[j] - GImgPtr0[j] + GImgPtr1[j1] - GImgPtr0[j1]) / 2.0;
            MagPtr[j] = sqrt(Dx * Dx + Dy * Dy);
            // 思路：如果abs(Dx)很小，那么是90度，如果abs(Dy)很小，那么是0度；
            // 如果Dx和Dy的模值差不多，同号则为45度，异号则为135度；
            // 如此可以避免多次调用atan2函数
            // 现在的问题是如何将这个界定写入代码之中
            // 而且也别忘了22.5度这种神奇的情况应作何判断
            double absDx = fabs(Dx);
            double absDy = fabs(Dy);
            if(absDx - absDy > 0.586 * absDx) // Dy相对于Dx来说很小，tan(22.5°)≈0.414
            {
                ThePtr[j] = 0;
            }
            else if(absDy - absDx > 0.586 * absDy)
            {
                ThePtr[j] = 90;
            }
            else if(Dx * Dy > 1e-6)
            {
                ThePtr[j] = 45;
            }
            else
            {
                ThePtr[j] = 135;
            }
        }
    }

    // 计算NMS
    // ChatGPT的意思是，只要看前后两个方向的点即可
    // 写个lambda省点空间
    // 这里，或者是上一步，过分简化了；所有网上的Canny算法帖子都有插值一说，我这里没有！！
    // 这可能是导致我的结果和老师的不一样的原因
    for(int i = 0; i < lena_height; i++)
    {
        uchar * ThetPtr = Theta.ptr<uchar>(i);
        double * MagniPtr = Magnitude.ptr<double>(i);
        for(int j = 0; j < lena_width; j++)
        {
            int dx1, dx2, dy1, dy2;
            switch (ThetPtr[j])
            {
            case 0:
                dy1 = std::max(0, j - 1);
                dy2 = std::min(lena_width - 1, j + 1);
                dx1 = i;
                dx2 = i;
                break;
            case 45:
                dy1 = std::min(lena_width - 1, j + 1);
                dy2 = std::max(0, j - 1);
                dx1 = std::max(0, i - 1);
                dx2 = std::min(lena_height - 1, i + 1);
                break;
            case 90:
                dx1 = std::max(0, i - 1);
                dx2 = std::min(lena_height - 1, i + 1);
                dy1 = j;
                dy2 = j;
                break;
            case 135:
                dy1 = std::max(0, j - 1);
                dy2 = std::min(lena_width - 1, j + 1);
                dx1 = std::max(0, i - 1);
                dx2 = std::min(lena_height - 1, i + 1);
                break;
            }
            double prev = Magnitude.at<double>(dx1, dy1);
            double post = Magnitude.at<double>(dx2, dy2);
            double point = Magnitude.at<double>(i, j);
            if(point < prev || point < post)
            {
                MagniPtr[j] = 0;
            }
        }
    }

    // 二值化与最终的边缘展示
    // ChatGPT建议把这个任务作为一个深度优先或广度优先的小任务
    // 思路：遍历所有的strong点，把周边的weak点全部变成strong点，再把新strong点遍历一遍，迭代
    // 问题是：如此这般，会不会只是把weak点全部变成strong点？
    // 孤立的weak点应该会在这个流程中被滤掉
    // 我需要两个容器，一个放旧strong点，一个放新strong点
    // 每次迭代中，旧strong点拿一个扔一个，迭代结束后新strong点全部填入旧strong点
    // 扔掉的点保留在cv::Mat中
    double MagMin;
    double MagMax;
    cv::Mat MagFlatten = Magnitude.reshape(0, 1);
    cv::minMaxLoc(MagFlatten, &MagMin, &MagMax);
    double HighThresh = 0.1 * MagMax; // not perfect, but it have to do now
    double LowThresh = 0.4 * HighThresh;

    cv::Mat HighMap(lena_height, lena_width, CV_8U);
    cv::Mat LowMap(lena_height, lena_width, CV_8U);
    cv::threshold(Magnitude, HighMap, HighThresh, 256, cv::THRESH_BINARY);
    cv::threshold(Magnitude, LowMap, LowThresh, 256, cv::THRESH_BINARY);

    std::queue<int> strongPoints;
    for(int i = 0; i < lena_height; i++)
    {
        uchar * HighPtr = HighMap.ptr<uchar>(i);
        for(int j = 0; j < lena_width; j++)
        {
            if(HighPtr[j] > 0)
            {
                strongPoints.push(i);
                strongPoints.push(j);
            }
        }
    }
    while(! strongPoints.empty())
    {
        int idx_i = strongPoints.front();
        strongPoints.pop();
        int idx_j = strongPoints.front();
        strongPoints.pop();

        uchar * HighMPtr = HighMap.ptr<uchar>(idx_i);
        HighMPtr[idx_j] = 255;
        uchar * LowMPtr = LowMap.ptr<uchar>(idx_i);
        LowMPtr[idx_j] = 0;
        for(int dx = -1; dx <= 1; dx++)
        {
            if(idx_i + dx < 0 || idx_i + dx >= lena_height)
            {
                continue;
            }
            uchar * LowPtr = LowMap.ptr<uchar>(idx_i + dx);
            for(int dy = -1; dy <= 1; dy++)
            {
                if(idx_j + dy < 0 || idx_j + dy >= lena_width)
                {
                    continue;
                }
                if(LowPtr[idx_j + dy] > 1e-6)
                {
                    strongPoints.push(idx_i + dx);
                    strongPoints.push(idx_j + dy);
                }
            }
        }
    }

    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_edges.jpg", HighMap);
    return 0;
}