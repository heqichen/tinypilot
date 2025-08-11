$fn=40;
epsilon = 0.3;


pcb_width = 89; // x-Axis
pcb_depth = 57; // y-Axis
pcb_height = 1.7; // z-Axis

module round_cube(length, width, height, dimeter) {
    r = dimeter / 2.0;
    hull() {
        translate([r, r, 0]) cylinder(height, r, r);
        translate([length-r, r, 0]) cylinder(height, r, r);
        translate([r, width-r, 0]) cylinder(height, r, r);
        translate([length-r, width-r, 0]) cylinder(height, r, r);
    }
}

module pcb_board() {
    color([0, 1, 0])
    translate([-epsilon, -epsilon, 0])
    round_cube(pcb_width + epsilon*2, pcb_depth+epsilon*2, pcb_height+epsilon, 4.0);    
}

module mock_cpu() {
    xlen = 22;
    ylen = 22;
    height = 1.3;
    xlen2 = 17.8;
    ylen2 = 17.8;
    height2 = 1.6;
    translate([47.87, 25.95, pcb_height])
    union() {
        translate([0, 0, height/2]) cube([xlen, ylen, height], true);
        translate([0, 0, height2/2]) cube([xlen2, ylen2, height2], true);
    }
}

module mounting_holes() {
    translate([23.5, 53.5, 0]) cylinder(10, 3/2, 3/2, true);
    translate([23.5, 4.5, 0]) cylinder(10, 3/2, 3/2, true);
    translate([81.5, 4.5, 0]) cylinder(10, 3/2, 3/2, true);
    translate([81.5, 53.5, 0]) cylinder(10, 3/2, 3/2, true);
}

module usb2() {
    width = 14; // epsilon included
    height = 14.5+epsilon*2;
    color([1, 1, 1])
    translate([-5, 10-width/2, pcb_height-epsilon]) cube([17.5, width, height]);
}

module usb3() {
    width = 14;  // epsilon included
    height = 14.5+epsilon*2;
    color([0, 0, 1])
    translate([-5, 27.87-width/2, pcb_height-epsilon]) cube([17.5, width, height]);
}

module ethernet() {
    width = 17;
    height = 13.5+epsilon*2;
    color([0.5, 0.5, 1])
    translate([-5, 46.86-width/2, pcb_height-epsilon]) cube([21, width, height]);
}

module usbc() {
    xlen = 3.15 + epsilon*2;
    ylen = 15;
    height = 9.0 + epsilon*2;
    
    color([1, 0, 0])
    translate([29.12 - xlen/2, pcb_depth-ylen+5, pcb_height])
    cube([xlen, ylen, height]);
}

module hdmi_out() {
    xlen = 15 + epsilon*2;
    ylen = 15;
    height = 6.4 + epsilon*2;
    color([0, 0, 0])
    translate([42.60-xlen/2, pcb_depth-ylen+5, pcb_height])cube([xlen, ylen, height]);
}


module audio() {
    dimeter = 5 + epsilon*2;
    height = 2.5; // No epsilon needed
    ylen = 15;
    translate([56.62, pcb_depth-ylen/2+5, dimeter/2+pcb_height])
    rotate([90, 0, 0])
    cylinder(ylen, dimeter/2, dimeter/2, true);
}

module hdmi_in() {
    xlen = 15 + epsilon*2;
    ylen = 15;
    height = 6.4 + epsilon*2;
    color([0, 0, 0])
    translate([70.60-xlen/2, pcb_depth-ylen+5, pcb_height])cube([xlen, ylen, height]);
}


module power_button() {
    dimeter = 2.2 + epsilon*2;
    height = 4.6/2; // No epsilon needed
    ylen = 15;
    color([0.9, 0.6, 0.2])
    translate([pcb_width-ylen/2+5, 49.07, pcb_height+height])
    rotate([0, 90, 0])
    cylinder(ylen, dimeter/2, dimeter/2, true);

}

// Buttom

module tf_card() {
    ylen = 11 + epsilon*2;
    xlen = 15;
    height = 2.5 +epsilon*2;
    
    translate([pcb_width-xlen+5, 14.42-ylen/2, -height+epsilon])cube([xlen, ylen, height]);
}

module nvme_storage() {
    
    xlen = 84 + epsilon*2;
    ylen = 22 + epsilon*2;
    height = 5+epsilon*2;
    
    translate([3.46-epsilon, 32.65-ylen/2, -height+epsilon])cube([xlen, ylen, height]);
}

module gpio() {
}

module antenna() {
}

module camera() {
}



difference() {
    union() {
        pcb_board();
        mock_cpu();
        usb2();
        usb3();
        ethernet();
        usbc();
        hdmi_out();
        audio();
        hdmi_in();
        power_button();
        // bottom
        tf_card();
        nvme_storage();
        
        gpio();
        antenna();
        camera();
    }
    mounting_holes();
}
