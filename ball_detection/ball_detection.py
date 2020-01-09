import myImageLibrary as ml
image_name = input('the name of the image contain balls want to detect:(in pgm format or ppm format)\n')
gs = False
if 'pgm' in image_name:
    gs = True
output_name = input('the name of the output highlighted image:\n')
downscale = input('what is the down scale ratio? (suggest > 10)\n')
lower_bound = input('what is the expected lower bound of the radius(the ratio to the longest side of the image), if unknown, using 0\n')
lower_bound = float(lower_bound)
lower_bound = min(0.5,max(0,lower_bound))
upper_bound = input('what is the expected lower bound of the radius(the ratio to the longest side of the image), if unknown, using 0.5\n')
upper_bound = float(upper_bound)
upper_bound = min(0.5,max(0,upper_bound))
threshold = input('what is minimum voting threshold? 1 will only highlight the circle with most vote, fraction number will highlight the circle greater than the threshold\n')
threshold = float(threshold)
threshold = min(1,max(threshold,0))
highlight_color = input('what is the highlighted color,0-255?\n')
highlight_color = int(highlight_color)
highlight_color = min(255,max(highlight_color,0))
hg = ml.Hough_Transformer(downscale=int(downscale))
hg.detect_circle(image_name,output_name, grayscale = gs, estimate_ratio =[lower_bound,upper_bound], support_threshold = threshold, highlight_width = 3, highlight = highlight_color)