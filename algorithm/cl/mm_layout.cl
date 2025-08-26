__kernel void add_one(__global int* data) {
    int id = get_global_id(0);
    data[id] += 1;
    data[id] -= 1;
    data[id] += 1;
    data[id] *= 50;
    data[id] /= 50;
}
