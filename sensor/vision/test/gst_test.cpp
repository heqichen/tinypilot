#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <iostream>

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // 构建pipeline
    std::string pipeline_desc = "filesrc location=mono_color.mp4 ! appsink name=sink";
    
    std::cout << "1" << std::endl;

    
    GError *error = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_desc.c_str(), &error);

    if (!pipeline) {
        std::cerr << "Failed to create pipeline: " << (error ? error->message : "unknown error") << std::endl;
        if (error) g_error_free(error);
        return -1;
    }
    
    std::cout << "2" << std::endl;

    // 获取appsink
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    gst_app_sink_set_emit_signals((GstAppSink*)appsink, false);
    gst_app_sink_set_drop((GstAppSink*)appsink, true);
    gst_app_sink_set_max_buffers((GstAppSink*)appsink, 1);

    // 启动pipeline
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    std::cout << "3" << std::endl;

    // 读取每一帧
    while (true) {
            
        std::cout << "7" << std::endl;
    
        GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
            
        std::cout << "6" << std::endl;

        if (!sample) {
            // 视频结束
            break;
        }
            
        std::cout << "5" << std::endl; 

        GstBuffer *buffer = gst_sample_get_buffer(sample);
        GstCaps *caps = gst_sample_get_caps(sample);
        GstStructure *s = gst_caps_get_structure(caps, 0);

        int width, height;
        gst_structure_get_int(s, "width", &width);
        gst_structure_get_int(s, "height", &height);
            
        std::cout << "4" << std::endl;

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            // 这里 map.data 指向一帧的RGB数据，长度为 map.size
            // 你可以在这里处理每一帧
            std::cout << "Got frame: " << width << " x " << height << ", data size: " << map.size << std::endl;

            gst_buffer_unmap(buffer, &map);
        }

        gst_sample_unref(sample);
    }

    // 清理
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);

    return 0;
}
