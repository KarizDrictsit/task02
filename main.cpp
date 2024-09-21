#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(){
    Mat img = imread("/home/kariz/opencv_project/resources/test_image.png");
    
    if(img.empty()){cout << "Not Found" << endl;
        return -1;
    }//检查是否打开;
    
    Mat grayImg,hsvImg,CannyImg,mediaBlurImg,GaussianBlurImg;
    
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    cvtColor(img, hsvImg, COLOR_BGR2HSV);

    Canny(grayImg, CannyImg, 50 , 150);

    blur(img, mediaBlurImg, Size(5, 5));

    GaussianBlur(img, GaussianBlurImg, Size(3, 3), 0);


  
    Mat M;
    Point2f Center(img.cols / 2.0, img.rows / 2.0);
    M = getRotationMatrix2D(Center, 35, 1.0);
    Mat xzImg;
    warpAffine(img, xzImg, M, img.size());

    Rect roi(0, 0, img.cols / 2, img.rows / 2);
    Mat cjImg = img(roi);

    //因为后续还要对img操作，这里先将旋转与裁剪结果输出一下
    imwrite("cj.png", cjImg);
    imwrite("xz.png", xzImg);

  
 
    //因为后续还要对img操作，这里把提取hinglight区域的代码放到前面
    Mat mask_hl;
    inRange(hsvImg, Scalar(0, 150, 50), Scalar(179, 255, 255), mask_hl);
    Mat hlImg;
    bitwise_and(img, img, hlImg, mask_hl);//提取高亮区域

    Mat hlgryImg, hlbnyImg, hldltImg, hlerdImg;
    //灰度化
    cvtColor(hlImg, hlgryImg, cv::COLOR_BGR2GRAY);
    //二值化
    threshold(hlImg, hlbnyImg, 100, 255, THRESH_BINARY);
    //膨胀
    dilate(hlImg, hldltImg, Mat(), Point(-1, -1), 2);
    //腐蚀
    erode(hlImg, hlerdImg, Mat(), Point(-1, -1), 2);

    //由于任务书中“漫水处理”这一环节的诠释并不清楚，我实现的是“对提取了高亮区域且进行了灰度化处理后的图像（即hlgryImg）进行漫水填充”这个操作。
    //创建一个副本以进行填充
    Mat copyImg = hlgryImg.clone();
    //设置种子点和填充颜色
    Point seedPoint(100, 100);//根据需要选择种子点
    Scalar newColor(255, 0, 0);
    //执行漫水填充
    floodFill(copyImg, seedPoint, newColor);

   
    //提取红色区域，其中包含两个过程，分别是 1.创建红色掩膜 以及 2.利用函数“按位与操作”将所用红色区域提取出来并储存之于 redImg 中
        //创建掩模
        Mat mask1, mask2, mask;
        inRange(hsvImg, Scalar(0, 100, 100), Scalar(10, 255, 255), mask1);
        inRange(hsvImg, Scalar(160, 100, 100), Scalar(180, 255, 255), mask2);
        mask = mask1 | mask2;
    //按位与操作
    Mat redImg;
    bitwise_and(img, hsvImg, redImg, mask);

    //后续还要对redImg进行操作，先输出一下，称作red
    imwrite("red.png", redImg);

    Mat red_gray_img, red_gray_canny_img;
    cvtColor(redImg, red_gray_img, COLOR_BGR2GRAY);
    Canny(red_gray_img, red_gray_canny_img, 100, 200);
    
    //查找轮廓(此处鉴于后面要计算面积，因此使用更精确的轮廓逼近方法)
    vector< vector<Point> > contours;
    findContours(red_gray_canny_img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    //请注意，findcontours函数的输入图像必须是canny图像，否则会产生“核心已转储”
 
    Mat contour_redImg = Mat::zeros(redImg.size(), CV_8UC3);

    // 遍历轮廓, 同时进行：计算轮廓的面积, 绘制轮廓
    double totalarea = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        // 计算轮廓的面积
        double area = contourArea(contours[i]);
        totalarea += area;
        drawContours(contour_redImg, contours, (int)i,  Scalar(0, 0, 255), 2);
        }
        cout << " Area: " << totalarea << std::endl;
    
    //分别绘制 红色区域的boundingbox  和  原图的 红色的boundingbox
    for (const auto& contour : contours) {
        Rect boundingBox = boundingRect(contour);
        rectangle(redImg, boundingBox, Scalar(0, 255, 0), 2); // 绿色框，线宽为2
    }
    for (const auto& contour : contours) {
        Rect boundingBox = boundingRect(contour);
        rectangle(img, boundingBox, Scalar(0, 0, 255), 2); // 红色框，线宽为2
    }

    //接下来开始绘制原图的红色的外轮廓，这里就用粗略的逼近方法寻找
    vector< vector<Point> > contours1;
    
    findContours(CannyImg, contours1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImg = Mat::zeros(img.size(), CV_8UC3);
    for (size_t i = 0; i < contours1.size(); i++) {
        drawContours(contourImg, contours1, (int)i, Scalar(0, 0, 255), 2); // 绘制轮廓
    }
    
    //我只是想表示我已经寻找出了红色区域的轮廓与boundingbox，为了方便，我不将其保存，只将其显示
    imshow("red_contours.png", contour_redImg);
    waitKey(0);
    
    imshow("red_boundingbox.png", redImg);
    waitKey(0);




    Mat kb = Mat::zeros(400, 400, CV_8UC3);

    //绘制一个圆形
    Point center(200, 200);
    int radius = 50;
    Scalar circleColor(0, 0, 255);
    circle(kb, center, radius, circleColor, -1); //填充

    //绘制一个方形
    Point topLeft(100, 100); 
    Point bottomRight(150, 150); 
    Scalar rectColor(255, 0, 0); 
    rectangle(kb, topLeft, bottomRight, rectColor, 1); 

    //绘制文字
    string text = "Fate/GrandOrder";
    Point textOrg(50, 350); // 文字位置
    Scalar textColor(255, 255, 255); // 白色
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 2;
    putText(kb, text, textOrg, fontFace, fontScale, textColor, thickness);

    //为了方便，这个空白的图像我不进行保存，只将它显示出来即可
    imshow("Shapes and Text", kb);
    waitKey(0);
 

    imwrite("gry.png", grayImg);
    imwrite("hsv.png", hsvImg);
    imwrite("Cny.png", CannyImg);
    imwrite("meB.png", mediaBlurImg);
    imwrite("GaB.png", GaussianBlurImg);
    imwrite("boundingbox.png", img);
    imwrite("contours.png", contourImg);
    imwrite("hlgry.png", hlgryImg);
    imwrite("hlbny.png", hlbnyImg);
    imwrite("hldlt.png", hldltImg);
    imwrite("hlerd.png", hlerdImg);
    imwrite("FloodFill.png", copyImg);



    return 0;
}
