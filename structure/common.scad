// $fn=40;
epsilon = 0.3;

// ------ parameters for main board
nvme_cetner_y = 32.65;
wall_thickness = 1.5;

step_height = 1;
step_thickness = 1;

pcb_width = 89; // x-Axis
pcb_depth = 57; // y-Axis
pcb_height = 1.7; // z-Axis

pcbMountingXPos = [23.5, 81.5];
pcbMountingYPos = [4.5, 53.5];

flange_height = 2.5;

cpu_pos_x = 47.87;
cpu_pos_y = 25.95;

top_cpu_sink_width = 20;

wall_height = pcb_height + step_height + flange_height;

tf_card_width = 11;
tf_card_pos_y = 14.42;



// ------ parameters for top fan
// no padding
top_fan_width = 20.3;
top_fan_depth = 20.3;
top_fan_height = 6.1;

// ------ parameters for top shell

top_shell_thickness = 3;
top_space_height = 12; // top of the PCB

top_fan_offset_x = top_cpu_sink_width/2 + top_fan_width/2 + 2 + 2;

// ------ parameters for bottom fan
bottom_fan_height = 6.3;

// ------ parameters for bottom shell
bottom_fan_center_x = 45;
bottom_fan_center_y = nvme_cetner_y;
bottom_shell_thickness = 3.0;
nvme_height = 5.0;
bottom_space_height = nvme_height + bottom_fan_height + 5; // 5mm space for air flow

module edge_step(height_offset=0, thickness_offset=0) {
    translate([
        -wall_thickness+step_thickness-thickness_offset, 
        -wall_thickness+step_thickness-thickness_offset, 
        wall_height-step_height-height_offset
    ])
    round_cube(
        pcb_width+wall_thickness*2-step_thickness*2 + thickness_offset*2, 
        pcb_depth+wall_thickness*2-step_thickness*2 + thickness_offset*2,
        step_height + height_offset*2,
        3
    );
}


module round_cube(length, width, height, dimeter, center=false) {
    r = dimeter / 2.0;
    
    tx = center ? -length / 2 : 0;
    ty = center ? -width / 2 : 0;
    
    translate([tx, ty, 0])
    hull() {
        translate([r, r, 0]) cylinder(height, r, r);
        translate([length-r, r, 0]) cylinder(height, r, r);
        translate([r, width-r, 0]) cylinder(height, r, r);
        translate([length-r, width-r, 0]) cylinder(height, r, r);
    }
}

