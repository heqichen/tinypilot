include <common.scad>

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
    translate([cpu_pos_x, cpu_pos_y, pcb_height])
    union() {
        translate([0, 0, height/2]) cube([xlen, ylen, height], true);
        translate([0, 0, height2/2]) cube([xlen2, ylen2, height2], true);
        
        // Heat sink
        color([0, 0, 0])
        translate([0, 0, 10/2 + height2]) cube([20, 20, 10], true);
    }
    
}

module mounting_holes() {
    for (x = pcbMountingXPos) {
        for (y = pcbMountingYPos) {
            translate([x, y, 0]) cylinder(10, 3/2, 3/2, true);
        }
    }
    
    /*
    translate([23.5, 53.5, 0]) cylinder(10, 3/2, 3/2, true);
    translate([23.5, 4.5, 0]) cylinder(10, 3/2, 3/2, true);
    translate([81.5, 4.5, 0]) cylinder(10, 3/2, 3/2, true);
    translate([81.5, 53.5, 0]) cylinder(10, 3/2, 3/2, true);
    */
}

module usb2() {
    length = 17.5 + 2;
    width = 14; // epsilon included
    height = 16+epsilon*2;
    color([1, 1, 1])
    translate([-2.5-epsilon-1, 10-width/2, pcb_height-epsilon]) cube([length, width, height]);
}

module usb3() {
    length = 17.5 + 2;
    width = 14;  // epsilon included
    height = 16 +epsilon*2;
    color([0, 0, 1])
    translate([-2.5-epsilon-1, 27.87-width/2, pcb_height-epsilon]) cube([length, width, height]);
}

module ethernet() {
    width = 17;
    length = 22 + 2;
    height = 14+epsilon*2;
    color([0.5, 0.5, 1])
    translate([-2.5-epsilon-1, 46.86-width/2, pcb_height-epsilon]) cube([length, width, height]);
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


module audio_jeck() {
    dimeter = 5 + epsilon*2;
    height = 2.5; // No epsilon needed
    ylen = 15;
    translate([56.62, pcb_depth-ylen/2+5, dimeter/2+pcb_height])
    rotate([90, 0, 0])
    cylinder(ylen, dimeter/2, dimeter/2, true);
}

module audio_body() {
    translate([56.62-7/2, pcb_depth-15, pcb_height])
    cube([7, 15, 6]);
}

module audio() {
    audio_jeck();
    audio_body();
}

module hdmi_in() {
    xlen = 15 + epsilon*2;
    ylen = 15;
    height = 6.4 + epsilon*2;
    color([0, 0, 0])
    translate([70.60-xlen/2, pcb_depth-ylen+5, pcb_height])cube([xlen, ylen, height]);
}

module mic() {
    mic_diameter = 4.5+epsilon*2;
    color([1, 1, 1])
    translate([pcb_width+epsilon - mic_diameter/2, pcb_depth+epsilon-mic_diameter/2, pcb_height])
    cylinder(3, mic_diameter/2, mic_diameter/2);
}

module fan_port() {
    color([0.1, 0.1, 0.1])
    translate([pcb_width+epsilon-3, 0, pcb_height]) cube([3, 6, 9]);
    
    xlen = 4.5 + epsilon*2;
    ylen = 7.5 + epsilon*2;
    h = 5.5 + epsilon;
    color([1, 1, 1])
    translate([pcb_width+epsilon-xlen, 11 - ylen/2, pcb_height]) cube([xlen, ylen, h]);
}

module power_button() {
    dimeter = 2.5 + epsilon*2;
    height = 4.6/2; // No epsilon needed
    ylen = 15;
    color([0.9, 0.6, 0.2])
    translate([pcb_width-ylen/2+5, 49.07, pcb_height+height])
    rotate([0, 90, 0])
    cylinder(ylen, dimeter/2, dimeter/2, true);

}

// Buttom

module tf_card() {
    ylen = tf_card_width + epsilon*2;
    xlen = 15;
    height = 2.5 +epsilon*2;
    
    translate([pcb_width-xlen+5, tf_card_pos_y-ylen/2, -height+epsilon])cube([xlen, ylen, height]);
}



module nvme_storage() {
    
    xlen = 84 + epsilon*2;
    ylen = 22 + epsilon*2;
    height = 5+epsilon*2;
    
    translate([3.46-epsilon, nvme_cetner_y-ylen/2, -height+epsilon])cube([xlen, ylen, height]);
}



module mock_mainboard() {
    difference() {
        union() {
            // top
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
            mic();
            fan_port();
            
            // bottom
            tf_card();
            nvme_storage();
            
            
            // gpio();
            // antenna();
            // camera();
        }
        mounting_holes();
    }
}

// mock_mainboard();


module fix_audio(top=true) {
    hull() {
        if (top) {
            translate([0, 0, -20]) audio_jeck();
        } else {
            translate([0, 0, 20]) audio_jeck();
        }
        audio_jeck();
    }
    
    hull() {
        if (top) {
            translate([0, 0, -20]) audio_body();
        } else {
            translate([0, 0, 20]) audio_body();
        }
        audio_body();
    }
}


module fix_tf_card(top = true) {
    hull() {
        if (top) {
            translate([0, 0, -20]) tf_card();
        } else {
            translate([0, 0, 20]) tf_card();
        }
        tf_card();
    }
}

module fix_power_button(top = true) {
    hull() {
        if (top) {
            translate([0, 0, -20]) power_button();
        } else {
            translate([0, 0, 20]) power_button();
        }
        power_button();
    }
}

module fix_pcb_board(top = true) {
    hull() {
        if (top) {
            translate([0, 0, -20]) pcb_board();
        } else {
            translate([0, 0, 20]) pcb_board();
        }
        pcb_board();
    }
}


// mock_mainboard();
