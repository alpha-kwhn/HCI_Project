#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace cv;
using namespace std;

void my_mouse_callback(int event, int x, int y, int flags, void* param);
CvRect box = cvRect(-1, -1, 0, 0);
bool drawing_box = false;

class bookNode
{
public:
    int           width, height;
    Mat           book;
    int           x, y;

public:
    bookNode(Mat _book, double w = 0.0, double h = 0.0, int _x = 0, int _y = 0) {
        book = _book;
        width = w;
        height = h;
        x = _x;
        y = _y;
    }
};

void draw_box(IplImage* img, CvRect rect) {
    cvRectangle(img, cvPoint(rect.x, rect.y),
        cvPoint(rect.x + rect.width, rect.y + rect.height),
        cvScalar(0xff, 0x00, 0x00));
}

int* bookcase_size(IplImage* image, int flag = 0) {
    IplImage* cp;
    if (flag == 0) {
        cp = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_8U, 3);
        cvResize(image, cp, CV_INTER_CUBIC);
    }
    else if (flag == 1) {
        cp = cvCreateImage(cvSize(720, 1280), IPL_DEPTH_8U, 3);
        cvResize(image, cp, CV_INTER_CUBIC);
    }
    else {
        cout << "Wrong bookcase_size flag!" << endl;
        return 0;
    }

    IplImage* temp = cvCloneImage(cp);
    cvNamedWindow("Box Example");
    cvSetMouseCallback("Box Example", my_mouse_callback, (void*)cp);
    while (1) {
        cvCopy(cp, temp);
        if (drawing_box) draw_box(temp, box);
        cvShowImage("Box Example", temp);
        if (cvWaitKey(15) == 27) break;
    }

    cvReleaseImage(&cp);
    cvReleaseImage(&temp);
    cvDestroyWindow("Box Example");

    int* case_size = new int[4];
    case_size[0] = box.x;
    case_size[1] = box.y;
    case_size[2] = box.width;
    case_size[3] = box.height;

    return case_size;
}

void my_mouse_callback(int event, int x, int y, int flags, void* param) {
    IplImage* image = (IplImage*)param;

    switch (event) {
    case CV_EVENT_MOUSEMOVE:
    {
        if (drawing_box) {
            box.width = x - box.x;
            box.height = y - box.y;
        }
    }
    break;
    case CV_EVENT_LBUTTONDOWN:
    {
        drawing_box = true;
        box = cvRect(x, y, 0, 0);
    }
    break;
    case CV_EVENT_LBUTTONUP:
    {
        drawing_box = false;
        if (box.width < 0) {
            box.x += box.width;
            box.width *= -1;
        }
        if (box.height < 0) {
            box.y += box.height;
            box.height *= -1;
        }
        draw_box(image, box);
    }
    break;
    default:
        break;
    }
}

void combination(vector<vector<int>>& test, vector<int>& comb, int n, int r, int k)
{
    if (comb.size() == r) {
        test.push_back(comb);
    }
    else if (k == n + 1)
    {

    }
    else {
        comb.push_back(k);
        combination(test, comb, n, r, k + 1);
        comb.pop_back();
        combination(test, comb, n, r, k + 1);
    }
}

int markDetect(Mat image, int flag = 0)
{
    Size patternSize(4, 4);
    vector<Point2f> corners;
    bool check;
    double lengthSum, lengthMean;
    int i, j;

    lengthSum = lengthMean = 0;

    if (flag == 0) {
        resize(image, image, Size(1280, 720));
    }
    else if (flag == 1) {
        resize(image, image, Size(720, 1280));
    }
    else {
        cout << "Wrong markDetect flag!" << endl;
        return -1;
    }

    Mat tmp_Gray;

    cvtColor(image, tmp_Gray, COLOR_BGR2GRAY);

    check = findChessboardCorners(tmp_Gray, patternSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

    if (check) {
        cornerSubPix(tmp_Gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

        for (i = 0; i < 4; i++) {
            for (j = 0; j < 3; j++) {
                lengthSum += sqrt(pow(corners.at(i * 4 + j).x - corners.at(i * 4 + j + 1).x, 2) + pow(corners.at(i * 4 + j).y - corners.at(i * 4 + j + 1).y, 2));
                lengthSum += sqrt(pow(corners.at(j * 4 + i).x - corners.at(j * 4 + i + 4).x, 2) + pow(corners.at(j * 4 + i).y - corners.at(j * 4 + i + 4).y, 2));
            }
        }
        lengthMean = lengthSum / 24;

        tmp_Gray.release();
        return (int)round(lengthMean);
    }
    else {
        cout << "mark cannot found!" << endl;
        tmp_Gray.release();
        return -1;
    }
}

vector<bookNode> cutByHeight(vector<Mat> image)
{
    vector<bookNode> rt_books;
    vector<int> line_y;
    int y_low = -1;

    for (int i = 0; i < image.size(); i++) {
        //Mat original_book; image[i].copyTo(original_book);
        Mat gray;
        cvtColor(image[i], gray, COLOR_BGR2GRAY);
        //imshow("gr_" + to_string(i), gray);

        Mat sobel1;
        Mat sobel2;
        Mat sobel;
        Sobel(gray, sobel1, CV_8U, 0, 1);
        rotate(gray, gray, ROTATE_180);
        Sobel(gray, sobel2, CV_8U, 0, 1);
        rotate(sobel2, sobel2, ROTATE_180);
        rotate(gray, gray, ROTATE_180);
        sobel = sobel1 + sobel2;
        //imshow("ddd" + to_string(i), sobel);
        //waitKey();

        Mat thresh;
        threshold(sobel, thresh, 100, 255, THRESH_BINARY);
        //imshow("thresh_" + to_string(i), thresh);
        //waitKey();
        Mat horizon = thresh.clone();
        Mat horizon_structure = getStructuringElement(MORPH_RECT, Size(horizon.cols / 3, 1));
        morphologyEx(horizon, horizon, MORPH_OPEN, horizon_structure);
        morphologyEx(horizon, horizon, MORPH_OPEN, horizon_structure);
        morphologyEx(horizon, horizon, MORPH_CLOSE, horizon_structure);
        //imshow("horizon_" + to_string(i), horizon);
        //waitKey();

        vector<Vec4i> linesP;
        HoughLinesP(horizon, linesP, 1, CV_PI / 100, image[i].cols / 4);

        Mat hough;
        image[i].copyTo(hough);

        for (size_t j = 0; j < linesP.size(); j++)
        {
            Vec4i l = linesP[j];
            line(hough, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255));
            //cout << i << ": (" << l[0] << ", " << l[1] << ") (" << l[2] << ", " << l[3] << ")" << endl;

            line_y.push_back(l[1]);
        }
        //imshow("hough" + to_string(i), hough);
        //waitKey();
        sort(line_y.begin(), line_y.end());

        if (i == 0) y_low = line_y[line_y.size() - 1];

        int y1 = line_y[0];
        int y2 = y_low;
        int x1 = 0;
        int x2 = image[i].cols;

        Mat cutted_book = image[i](Range(y1, y2), Range(x1, x2));
        int width_book = x2 - x1;
        int height_book = y2 - y1;
        //imshow(to_string(i), cutted_book);

        rt_books.push_back(bookNode(cutted_book, width_book, height_book));

        line_y.clear();
    }

    return rt_books;

}

vector<bookNode> bookDetect(Mat image, int bookcase_cm)
{
    Mat book = image;
    int tmp = 0;

    Mat gray;
    cvtColor(book, gray, COLOR_BGR2GRAY);
    rotate(gray, gray, ROTATE_90_CLOCKWISE);
    rotate(book, book, ROTATE_90_CLOCKWISE);
    resize(gray, gray, Size(1280, 720));
    resize(book, book, Size(1280, 720));
    //imshow("gr", gray);

    Mat sobel1;
    Mat sobel2;
    Mat sobel;
    Sobel(gray, sobel1, CV_8U, 1, 0);
    rotate(gray, gray, ROTATE_180);
    Sobel(gray, sobel2, CV_8U, 1, 0);
    rotate(sobel2, sobel2, ROTATE_180);
    rotate(gray, gray, ROTATE_180);
    sobel = sobel1 + sobel2;
    //imshow("sobel", sobel);

    Mat thresh;
    threshold(sobel, thresh, 180, 255, THRESH_BINARY);
    //imshow("thresh", thresh);

    Mat canny;
    Canny(thresh, canny, 30, 100);
    //imshow("canny", canny);

    Mat opening;
    Mat kernel1 = getStructuringElement(MORPH_RECT, Size(1, 11), Point(-1, -1));
    morphologyEx(thresh, opening, MORPH_OPEN, kernel1);
    //imshow("opening", opening);

    Mat opening2;
    Mat kernel2 = getStructuringElement(MORPH_RECT, Size(1, 31), Point(-1, -1));
    morphologyEx(opening, opening2, MORPH_OPEN, kernel2);
    //imshow("opening2", opening2);

    Mat closing;
    Mat kernel3 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
    morphologyEx(opening2, closing, MORPH_CLOSE, kernel3);
    //imshow("closing", closing);

    vector<Vec2f> lines;
    HoughLines(closing, lines, 1, CV_PI / 100, 150);

    vector<Mat> books;
    vector<int> line_x;

    Mat img_hough;
    gray.copyTo(img_hough);

    Mat img_line;
    threshold(closing, img_line, 150, 255, THRESH_MASK);

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        if (b > 0.01) continue;
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(img_hough, pt1, pt2, Scalar(0, 0, 255), 2, 8);
        line(img_line, pt1, pt2, Scalar::all(255), 1, 8);

        line_x.push_back((pt1.x + pt2.x) / 2);
    }
    //imshow("img_hough", img_hough);
    //imshow("img_line", img_line);

    sort(line_x.begin(), line_x.end());

    for (size_t i = 0; i < line_x.size(); i++) {
        if (i == 0) {
            tmp = line_x.at(i);
        }
        if (i > 0) {
            if (tmp + bookcase_cm <= line_x.at(i)) {
                books.push_back(book(Range(0, 720), Range(tmp, line_x.at(i))));
                tmp = line_x.at(i);
            }
        }
    }

    //for (size_t i = 0; i < books.size(); i++) {
    //   imshow("img" + to_string(i), books.at(i));
    //}

    return cutByHeight(books);

}

bool compareWidth(const bookNode& a, const bookNode& b)
{
    return a.width < b.width;
}

bool compareHeight(const bookNode& a, const bookNode& b)
{
    return a.height > b.height;
}

void exceedSize(vector<bookNode>& bks, int bc_width, int bc_height)
{
    int num = bks.size();
    for (int i = 0; i < num; i++) {
        if (bks.at(i).width > bc_width || bks.at(i).height > bc_height) {
            bks.erase(bks.begin() + (i));
            i--;
            num--;

        }
    }

}

int main()
{
    IplImage* Ipl_image = cvLoadImage("image\\B_01.jpg", CV_LOAD_IMAGE_COLOR);
    Mat              bookcase = cvarrToMat(Ipl_image);
    Mat              book = imread("image\\A_01.jpg");
    Mat              bookList;
    Mat              imageROI;
    vector<bookNode> books;
    vector<bookNode> books_final;
    vector<bookNode> books_left;
    int              books_number;
    int              books_width_sum;
    int              bookcase_x;
    int              bookcase_y;
    int              bookcase_width;
    int              bookcase_height;
    int              bookList_width;
    int              bookList_height;
    int              bookcase_cm;
    int              book_cm;
    int              margin;
    int              cur_x;
    int              cur_y;
    int* bookcase_Size = new int[4];
    int             bookChecker;
    vector<vector<int>> comb;

    cout << "책을 넣고싶은 책장의 영역을 마우스로 드래그하여 표시하여 주십시오." << endl << "원하는 영역을 선택후 ESC를 눌러 다음단계로 진행해 주십시오" << endl;
    bookcase_Size = bookcase_size(Ipl_image, 0);
    system("cls");

    cout << "마커와 책을 검출 중입니다." << endl;
    // mark detection 
    bookcase_cm = markDetect(bookcase, 0);
    book_cm = markDetect(book, 1);

    // slice books 
    books = bookDetect(book, bookcase_cm);

    system("cls");
    bookList_width = 20;
    bookList_height = 0;
    for (auto i : books) {
        bookList_width += i.width + 20;
        if (bookList_height < i.height)
        {
            bookList_height = i.height;
        }
    }
    bookList_height += 40;

    bookList = Mat(Size(bookList_width, bookList_height), CV_8UC3, Scalar(0, 0, 0));
    cur_x = 20;
    cur_y = 20;
    for (int i = 0; i < books.size(); i++)
    {
        imageROI = bookList(Rect(cur_x, cur_y, books[i].width, books[i].height));
        books[i].book.copyTo(imageROI, books[i].book);
        cur_x += books[i].width + 20;
    }

    imshow("Book_List", bookList);
    cout << "정리하려는 책이 모두 인식되었는지 확인하여주십시오." << endl << "모두 인식되었으면 Y, 빠지거나 인식이 제대로 되지않은 책이 있다면 N을 입력하여 주십시오." << endl << "Y / N";
    while (true) {
        bookChecker = waitKey();
        if (bookChecker == 89 || bookChecker == 121)
        {
            break;
        }
        else if (bookChecker == 78 || bookChecker == 110)
        {
            system("cls");
            cout << "사용자 의견 : 책이 제대로 인식되지 않음" << endl;
            exit(1);
        }
        else
        {
            system("cls");
            cout << "잘못된 키입력" << endl << "책이 모두 인식되었으면 Y, 빠지거나 인식이 제대로 되지않은 책이 있다면 N을 입력하여 주십시오." << endl << "Y / N";
        }
    }
    system("cls");
    destroyWindow("Book_List");
    cout << "책을 책장에 맞게 끼우는 중입니다." << endl;

    // resize bookcase & books 
    if (bookcase_cm != -1 && book_cm != -1)
    {
        double calc = (double)book_cm / bookcase_cm;

        resize(bookcase, bookcase, Size((int)round(1280 * calc), (int)round(720 * calc)));
        rotate(book, book, ROTATE_90_CLOCKWISE);
        resize(book, book, Size(1280, 720));
        bookcase_x = (int)round(bookcase_Size[0] * calc);
        bookcase_y = (int)round(bookcase_Size[1] * calc);
        bookcase_width = (int)round(bookcase_Size[2] * calc);
        bookcase_height = (int)round(bookcase_Size[3] * calc);

        //imshow("bookcase", bookcase);
        //imshow("book", book);
    }
    else
    {
        system("cls");
        cout << "마커가 제대로 인식되지 않았습니다." << endl;
        exit(1);
    }

    // sort by width (ascending order)
    sort(books.begin(), books.end(), compareWidth);

    // exception: when width or height exceeds size of bookcase
    exceedSize(books, bookcase_width, bookcase_height);

    // get combinations of number of books
    books_number = books.size();
    vector<int> tmp_comb;
    for (int i = 1; i < books_number; i++) {
        combination(comb, tmp_comb, books_number - 1, i, 0);
    }

    // get final books

    margin = 2 * book_cm;
    books_width_sum = 0;
    // 1. get sum of width of books
    for (auto i : books) {
        books_width_sum += i.width;
    }

    // 2. check margin to all books
    bool flag = false;
    if (books_width_sum <= bookcase_width) {
        if (bookcase_width - books_width_sum >= margin) {
            flag = true;
        }
        else {
            for (auto i : books) {
                if (bookcase_height - i.height >= margin) {
                    flag = true;
                }
            }
        }
    }
    if (flag) {
        for (int j = 0; j < books.size(); j++) {
            books_final.push_back(books[j]);
        }
    }

    int tmp_books_sum, tmp_cnt;

    for (int i = 0; i < comb.size() && !flag; i++) {
        tmp_cnt = 0;
        tmp_books_sum = 0;

        for (int j = 0; j < books.size(); j++) {
            if (j == comb[i][tmp_cnt]) {
                if (tmp_cnt < comb[i].size() - 1)
                    tmp_cnt++;
                continue;
            }
            tmp_books_sum += books[j].width;
        }

        tmp_cnt = 0;

        if (tmp_books_sum <= bookcase_width) {
            if (bookcase_width - tmp_books_sum >= margin) {
                flag = true;
            }
            else {
                for (int j = 0; j < books.size(); j++) {
                    if (j == comb[i][tmp_cnt]) {
                        if (tmp_cnt < comb[i].size() - 1)
                            tmp_cnt++;
                        continue;
                    }

                    if (bookcase_height - books[j].height > margin) {
                        flag = true;
                    }
                }
            }

            if (flag) {
                for (int j = 0, k = 0; j < books.size(); j++) {
                    if (j == comb[i][k]) {
                        books_left.push_back(books[j]);
                        if (k < comb[i].size() - 1)
                            k++;
                    }
                    else {
                        books_final.push_back(books[j]);
                    }
                }
            }
        }
    }

    // sort by height (descending order)
    sort(books_final.begin(), books_final.end(), compareHeight);

    system("cls");
    cout << "책이 정리되었습니다." << endl;
    cur_x = bookcase_x;
    cur_y = bookcase_y;

    for (int i = 0; i < books_final.size(); i++) {
        cur_y += bookcase_height - books_final[i].height;
        imageROI = bookcase(Rect(cur_x, cur_y, books_final[i].width, books_final[i].height));
        books_final[i].book.copyTo(imageROI, books_final[i].book);
        cur_x += books_final[i].width;
        cur_y = bookcase_y;
    }

    int books_left_width = 40;
    for (auto i : books_left) books_left_width += i.width + 20;

    bookList = Mat(Size(books_left_width, bookList_height), CV_8UC3, Scalar(0, 0, 0));
    cur_x = 20;
    cur_y = 20;
    for (int i = 0; i < books_left.size(); i++) {
        imageROI = bookList(Rect(cur_x, cur_y, books_left[i].width, books_left[i].height));
        books_left[i].book.copyTo(imageROI, books_left[i].book);
        cur_x += books_left[i].width + 20;
    }

    imshow("Final", bookcase);
    imshow("Left Books", bookList);

    waitKey();
    return 0;
}