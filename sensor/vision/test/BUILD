cc_binary(
    name = "gst_test",
    srcs = ["gst_test.cpp"],
    copts = [
        "-I/usr/include/gstreamer-1.0",
        "-I/usr/include/glib-2.0",
        "-I/usr/lib/x86_64-linux-gnu/glib-2.0/include",
        "-pthread",
        "-I/usr/include/x86_64-linux-gnu"],
)

# include path from  `pkg-config --cflags gstreamer-1.0`