#include <bits/stdc++.h>
#include <torch/extension.h>
using namespace std;


int dcmp(float x){
    if (x > 1e-8)
        return 1;
    return x < -1e-8 ? -1 : 0;
}

struct Point
{
    float x, y;
};
float cross(Point a,Point b,Point c) ///叉积
{
    return (a.x-c.x)*(b.y-c.y)-(b.x-c.x)*(a.y-c.y);
}
Point intersection(Point a,Point b,Point c,Point d)
{
    Point p = a;
    float t =((a.x-c.x)*(c.y-d.y)-(a.y-c.y)*(c.x-d.x))/((a.x-b.x)*(c.y-d.y)-(a.y-b.y)*(c.x-d.x));
    p.x +=(b.x-a.x)*t;
    p.y +=(b.y-a.y)*t;
    return p;
}
//计算多边形面积
float PolygonArea(Point *p, int n)
{
    if(n < 3) return 0.0;
    float s = p[0].y * (p[n - 1].x - p[1].x);
    p[n] = p[0];
    for(int i = 1; i < n; ++ i)
        s += p[i].y * (p[i - 1].x - p[i + 1].x);
    return fabs(s * 0.5);
}

float CPIA(Point *a, Point *b, int na, int nb)//ConvexPolygonIntersectArea
{
    int cap = max(na,nb);
    Point p[cap * 2], tmp[cap * 2];
    int tn, sflag, eflag;
    a[na] = a[0], b[nb] = b[0];
    memcpy(p,b,sizeof(Point)*(nb + 1));
    for(int i = 0; i < na && nb > 2; i++)
    {
        sflag = dcmp(cross(a[i + 1], p[0],a[i]));
        for(int j = tn = 0; j < nb; j++, sflag = eflag)
        {
            if(sflag>=0) tmp[tn++] = p[j];
            eflag = dcmp(cross(a[i + 1], p[j + 1],a[i]));
            if((sflag ^ eflag) == -2)
                tmp[tn++] = intersection(a[i], a[i + 1], p[j], p[j + 1]); ///求交点
        }
        memcpy(p, tmp, sizeof(Point) * tn);
        nb = tn, p[nb] = p[0];
    }
    if(nb < 3) return 0.0;
    return PolygonArea(p, nb);
}

float SPIA(Point *a, Point *b, int na, int nb)///SimplePolygonIntersectArea 调用此函数
{
    int i, j;
    Point t1[4], t2[4];
    float res = 0, num1, num2;
    a[na] = t1[0] = a[0], b[nb] = t2[0] = b[0];
    for(i = 2; i < na; i++)
    {
        t1[1] = a[i-1], t1[2] = a[i];
        num1 = dcmp(cross(t1[1], t1[2],t1[0]));
        if(num1 < 0) swap(t1[1], t1[2]);
        for(j = 2; j < nb; j++)
        {
            t2[1] = b[j - 1], t2[2] = b[j];
            num2 = dcmp(cross(t2[1], t2[2],t2[0]));
            if(num2 < 0) swap(t2[1], t2[2]);
            res += CPIA(t1, t2, 3, 3) * num1 * num2;
        }
    }
    return fabs(res);//res为两凸多边形的交的面积 
}

bool sigmod(int i,int j, float * minx, float * miny, float * maxx, float * maxy) {
    float left1 = minx[i];
    float top1 = maxy[i];
    float right1 = maxx[i];
    float bottom1 = miny[i];
    float left2 = minx[j];
    float top2 = maxy[j];
    float right2 = maxx[j];
    float bottom2 = miny[j];
    return !( ((right1 < left2) || (bottom1 > top2)) || 
                ((right2 < left1) || (bottom2 > top1)) 
                );
}
bool sigmod_inter(int i, float * minx, float * miny, float * maxx, float * maxy, Point *a){
    float left1 = minx[i];
    float top1 = maxy[i];
    float right1 = maxx[i];
    float bottom1 = miny[i];
    float left2 = a[0].x;
    float top2 = a[0].y;
    float right2 = a[2].x;
    float bottom2 = a[2].y;
    return !( ((right1 < left2) || (bottom1 > top2)) || 
                ((right2 < left1) || (bottom2 > top1)) 
                );
}

torch::Tensor inter(const torch::Tensor& gts, const torch::Tensor& anchors){
    ///input: N * 24, M * 4
    /// output:overlaps[N][N]
    AT_ASSERTM(!gts.type().is_cuda(), "gts must be a CPU tensor");
    AT_ASSERTM(!anchors.type().is_cuda(), "anchors must be a CPU tensor");
    if (gts.numel() == 0 || anchors.numel() == 0) {
        return torch::empty({0}, gts.options().dtype(torch::kLong).device(torch::kCPU));
    }
    auto num_points_gt = gts.size(1) / 2;
    auto N = gts.size(0);
    auto M = anchors.size(0);
    torch::Tensor inters = torch::zeros({N, M});
    float minx[N + 100];
    float miny[N + 100];
    float maxx[N + 100];
    float maxy[N + 100];
    auto gt_boxes = gts.accessor<float,2>();
    auto anchor_boxes = anchors.accessor<float,2>();
//     cout << "woc box" << endl;
    Point poly_gts[N + 100][12];
    for(int j=0; j<N; j++) {
        minx[j] = 999999;
        miny[j] = 999999;
        maxx[j] = -1.0;
        maxy[j] = -1.0;
        for(int i=0;i<num_points_gt;i++) {
            poly_gts[j][i].x = gt_boxes[j][i*2];
            poly_gts[j][i].y = gt_boxes[j][i*2 + 1];
//             cout << poly_gts[j][i].x << " " << poly_gts[j][i].y << endl;
            minx[j] = std::min(minx[j], gt_boxes[j][i*2]);
            maxx[j] = std::max(maxx[j], gt_boxes[j][i*2]);
            miny[j] = std::min(miny[j], gt_boxes[j][i*2+1]);
            maxy[j] = std::max(maxy[j], gt_boxes[j][i*2+1]);
        }
    }
    
//     cout << "woc  gt" << endl;
    //cout << minx[0] << " " << maxx[0] << " " << miny[0] << " " << maxy[0] << endl;
    Point poly_anchors[M + 100][4];
    for(int j=0; j<M; j++) {
        poly_anchors[j][0].x = anchor_boxes[j][0];
        poly_anchors[j][0].y = anchor_boxes[j][1];
        poly_anchors[j][1].x = anchor_boxes[j][2];
        poly_anchors[j][1].y = anchor_boxes[j][1];
        poly_anchors[j][2].x = anchor_boxes[j][2];
        poly_anchors[j][2].y = anchor_boxes[j][3];
        poly_anchors[j][3].x = anchor_boxes[j][0];
        poly_anchors[j][3].y = anchor_boxes[j][3];
    }
    for(int i=0;i<N;i++) {
        for(int j=0;j<M;j++) {
            if(sigmod_inter(i,minx, miny, maxx, maxy, poly_anchors[j]) == false) {
                inters[i][j] = 0.0;
                continue;
            }
            float interArea = SPIA(poly_gts[i], poly_anchors[j], num_points_gt, 4);
            float area = fabs((anchor_boxes[j][0] - anchor_boxes[j][2]) * (anchor_boxes[j][1] - anchor_boxes[j][3]));
            float intersection = interArea / area;
            //cout << "woc:::" << interArea << " " << area << endl;
            inters[i][j] = (float) fabs(intersection);
        }
    }
    return inters;
}


torch::Tensor olp(const torch::Tensor& dets){
    ///input: N * 24
    /// output:overlaps[N][N]
    AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
    if (dets.numel() == 0) {
        return torch::empty({0}, dets.options().dtype(torch::kLong).device(torch::kCPU));
    }
    auto num_points = dets.size(1) / 2;
    auto N = dets.size(0);
    float minx[N + 100];
    float miny[N + 100];
    float maxx[N + 100];
    float maxy[N + 100];
    torch::Tensor overlaps = torch::zeros({N, N});
    auto boxes = dets.accessor<float,2>();

    Point polygons[N + 100][num_points * 2];
    for(int j = 0; j < N; j++) {
//         polygons[j] = new Point[num_points];
        minx[j] = 999999; 
        miny[j] = 999999;
        maxx[j] = -1.0; 
        maxy[j] = -1.0;
        for(int i = 0;i < num_points ; i++) {
            polygons[j][i].x = boxes[j][i*2];
            polygons[j][i].y = boxes[j][i*2 + 1];
            minx[j] = std::min(minx[j], boxes[j][i*2]);
            maxx[j] = std::max(maxx[j], boxes[j][i*2]);
            miny[j] = std::min(miny[j], boxes[j][i*2+1]);
            maxy[j] = std::max(maxy[j], boxes[j][i*2+1]);
        }
    }
    for(int i=0;i<N;i++) {
        for(int j=i;j<N;j++) {
            if(i == j){
                overlaps[i][j] = 1.0;
                overlaps[j][i] = 1.0;
                continue;
            }
            if(sigmod(i, j, minx, miny, maxx, maxy) == false) {
                overlaps[i][j] = 0.0;
                overlaps[j][i] = 0.0;
                continue;
            }
            float interArea = SPIA(polygons[i], polygons[j], num_points, num_points);
            float area1 = PolygonArea(polygons[i], num_points);
            float area2 = PolygonArea(polygons[j], num_points);
            //cout << area1 << " " << area2 << " " << interArea << endl;
            float IoU = interArea / (area1 + area2 - interArea);
            overlaps[i][j] = IoU;
            overlaps[j][i] = IoU;
        }
    }
//     for(int i=0;i<N;i++) {
//         delete [] polygons[i];
//     }
//     delete [] polygons;
//     delete [] minx;
//     delete [] miny;
//     delete [] maxx;
//     delete [] maxy;
    return overlaps;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("olp", &olp, "polygon overlaps");
  m.def("inter", &inter, "polygon intersections");
}

