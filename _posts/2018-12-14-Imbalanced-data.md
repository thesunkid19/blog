---
layout: post
title: "[Deep learning] Xử lý vấn đề dữ liệu hình ảnh bị chênh lệch về số lượng giữa các class"
date: 2018-12-15
categories: blog
tags: [DL,CV]
---

Vấn đề chênh lệch về số lượng trainning samples giữa các class là một vấn đề thường xuyên gặp phải trong thực tế và gây ra nhiều khó khăn cho quá trình trainning classifer để tạo ra một bộ phân loại robust. 

Chẳng hạn ta muốn trainning 1 model classifer hình ảnh cho xe ô tô tự lái để phân biệt giữa đoạn đường không có vật cản và đoạn đường có vật cản (giả sử là đàn bò). Sử dụng 1 camera từ ô tô để ghi lại tập dữ liệu cho việc trainning thì ta có thể thu được 1 tập frames với hầu hết các frame ảnh ghi lại hình ảnh của đoạn đường không có vật cản, và rất ít mẫu hình ảnh đàn bò đi ngang qua đường. Nếu chỉ đơn thuần đem bộ dữ liệu này đi gán nhãn và train, khả năng nhận biết được vật cản của model thu được sẽ là rất kém.

Ở ví dụ này, ta có thể nhận ra ngay một giải pháp đơn giản là đi thu thập thêm mẫu, tuy nhiên có rất nhiều bài toán khác trong thực tế mà việc đi thu thập thêm mẫu cho class quan trọng là rất khó khăn (một ví dụ là trong dữ liệu y tế). Bài viết này sẽ giới thiệu một số phương pháp hướng tới việc giúp giải quyết tốt hơn cho bài toán có ràng buộc này. (*Lưu ý*: ví dụ được lấy là binary class chỉ để tiện cho việc hình dung, nội dung của bài viết có bao gồm cả các bài toán multi-class, multi-lable)

Điểm bất ngờ là khi bài toán này giải quyết được hiệu quả, các thuật toán object detector one-stage với tốc độ xử lý nhanh chóng có thể đạt stage-of-the-art về độ chính xác so với các thuật toán two-stage (RCNN,...), một task dường như có nhiều điểm khác biệt so với classify.

### Sử dụng metric phù hợp
Việc dựa vào Accuracy của quá trình trainning để đánh giá mức độ chính xác của thuật toán trong trường hợp này là không phù hợp. F1-score, mặc dù không phải là một metric sử dụng để optimize (có thể dùng soft F1-score), nhưng là một metric tốt dùng cho việc đánh giá model trong trường hợp này, đặc biệt là Macro-F1 (thang F1 trung bình cộng tất cả F1 của nhiều class trong bài toán multiclass). 

![](https://raw.githubusercontent.com/thesunkid19/blog/gh-pages/img/Precisionrecall.svg.png)

Ngoài ra còn một số metric khác phù hợp cho bài toán này [[1]](https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/)

### Oversampling data
Bằng cách duplicate nhiều lần dữ liệu hình ảnh của class có số mẫu nhỏ, ta được 1 bộ data có số mẫu giữa các class cân bằng hơn. Trong cuộc thi Zalo AI, vấn đề imbalanced data được nhiều nhóm giải quyết bằng giải pháp oversampling (upsampling) [[2]](https://www.facebook.com/groups/machinelearningcoban/permalink/551444705312941/).
Tuy nhiên, cần thận trọng trong số lần duplicate mẫu, vì bản chất model CNN sau khi trainning sẽ giữ lại các invariant features nhận được từ data trainning, việc 1 bức ảnh xuất hiện quá nhiều lần sẽ khiến model bị overfitting với các feature ở bức ảnh đó. Có thể thấy như trong giải pháp của đội tham gia Zalo AI trên, tỷ lệ upsampling data được tune khá là công phu. 

Ngoài ra, ta còn có thể oversampling bằng cách augmentation data cho các lớp có số lượng mẫu ít, trong khi không apply điều này cho các lớp còn lại.
![](https://raw.githubusercontent.com/thesunkid19/blog/gh-pages/img/oversampling.png)

### Undersampling data
Tương tự như ý tưởng oversampling, phương pháp undersampling chỉ thực hiện ở giai đoạn preprocessing. Thay vì duplicate các mẫu ở lớp yếu, ta loại bỏ đi bớt mẫu ở các lớp có nhiều mẫu đi để có được một bộ data cân bằng.
Giải pháp này chỉ hiệu quả trong trường hợp số lượng data là rất lớn so với model. Chẳng hạn như ta có quá nhiều hình ảnh ở các đoạn đường không có vật cản, khi đó ta có thể lọc bỏ bớt đi các mẫu này trước khi đưa vào traninng.

### Weighted loss
Cross-Entropy loss là một hàm loss thông thường được sử dụng trong các bài toán classify. Giải sử ta có bài toán classify như đã được nêu trước đó. Khi ta sử dụng Cross entropy loss cho bài toán này, thì loss ở 1 mẫu thuộc lớp t sẽ có công thức: 
$\mathrm { CE } \left( p _ { \mathrm { t } } \right) = - \log \left( p _ { \mathrm { t } } \right)$

Ở phương pháp này, ta sử dụng một làm loss khác tinh chỉnh dựa trên C-E để giải quyết vấn đề không cân xứng trong dữ liệu. Với ý tưởng chủ đạo là thêm vào một trọng số khác nhau cho các lớp vào loss:
$\mathrm { WCE } \left( p _ { \mathrm { t } } \right) = - \alpha _ { \mathrm { t } } \log \left( p _ { \mathrm { t } } \right)$

Giải sử trong bài toán này, ta chọn α0 = 0.25 (lớp không vật cản),  a1 = 0.75 (lớp có vật cản) thì hàm loss sẽ đặt nặng hơn việc sai lệch trong quá trình train của lớp có vật cản. Từ đó sẽ điều chỉnh weight của model nhiều hơn theo lớp này.

Nhìn về mặt giá trị hàm loss, việc áp đặt thêm hệ số này khá giống với việc upsampling, bằng cách ta duplicate thêm mẫu cho lớp yếu thì cũng tương tự như việc tăng hệ số loss của lớp này, tuy nhiên khi train theo mini-batch thì tác động của 2 phương pháp này là khác nhau.

### Focal loss
Đây là hàm loss được sử dụng trong mạng object-detection RetinaNet của Facebook [[3]](https://arxiv.org/abs/1708.02002). Hàm loss này giúp giải quyết vấn đề là điểm yếu của các thuật toán object-detection 1-stage, giúp tăng độ chính xác của những mô hình mang sẵn trong mình ưu thế là thời gian inference rất nhanh. Trước tiên hãy cùng tìm hiểu xem hàm loss này hoạt động như thế nào.

Focal loss thực tế cũng là một làm loss tinh chỉnh từ C-E, với hyperparameter mới là γ gọi là tham số tập trung (focusing parameter):

$\mathrm { FL } \left( p _ { t } \right) = - \left( 1 - p _ { t } \right) ^ { \gamma } \log \left( p _ { t } \right)$

Có thể hiểu Focal loss hoạt động như một Weighted C-E loss, tuy nhiên hệ số log của loss này không phải là một tham số cố định đặt ra nhằm balance lại số mẫu dữ liệu như WCE, hệ số của Focal loss có thể được hiểu là tự điều chỉnh (adaptive parameter) theo độ chính xác của xác suất dự đoán từ model (pt).

Ví dụ: khi pt của 1 mẫu đang train thuộc lớp t là nhỏ, FL(pt) sẽ không thay đổi quá nhiều. (chẳng hạn chọn γ = 1). Nhưng khi pt = 0.9, khi đó hệ số của hàm log này gần như bằng 0, model sẽ ít quan tâm hơn tới việc twist tham số theo mẫu này.

Theo như bài báo gốc trình bày về hàm loss này, ý nghĩa của hàm loss này không chỉ để giải quyết vấn đề giữa dominate class và minor class do imbalanced mà còn tập trung vào các đối tượng là "hard case" bằng cách giảm hệ số của "easy case". Khi mẫu đã có pt cao, ta ít quan tâm hơn tới feature ở mẫu này để tập trung twist tham số theo những mẫu khó nhận biết.

**Tham khảo**: \\
[1] [Classification Accuracy is Not Enough: More Performance Measures You Can Use](https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/)\\
[2][Nhóm công khai Forum Machine Learning cơ bản | Facebook](https://www.facebook.com/groups/machinelearningcoban/permalink/551444705312941/)\\
[3][1708.02002 Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)\\
[4][Focal Loss Demystified – Escapades in Machine Learning – Medium](https://medium.com/adventures-with-deep-learning/focal-loss-demystified-c529277052de)\\
[5][Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, FPN, RetinaNet and…](https://medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359)\\
[6][The intuition behind RetinaNet – Prakash Jay – Medium](https://medium.com/@14prakash/the-intuition-behind-retinanet-eb636755607d)
