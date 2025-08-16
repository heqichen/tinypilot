$fn=40;

include <mainboard.scad>
include <top_fan.scad>




module b() {include <bottom_shell.scad>} //b();


/**

- [x] 1. space for board // smaller than PCB
- [x] 2. wall for PCB // to bottom flange
- [x] 4. fan
- [x] 5. all interfaces
- [x] 6. tf card / x audio / x button
- [x] 7. screws for board
- [x] 8. screws for box
- [x] 9. edge with top, step fitting
- [x] 10. airflow route
- [x] 11. fan port / x mic too close to edge, need fix
- [x]12, gpio, atenna, camera
12, $fn


**/


step_height = 1;

top_padding = 1;

module outline() {
    
    max_y = pcb_height + top_space_height + top_shell_thickness;
    min_y = pcb_height + flange_height;
    entity_height = max_y - min_y;
    
    
    translate([-wall_thickness, -wall_thickness, min_y])
    round_cube(pcb_width+wall_thickness*2, pcb_depth+wall_thickness*2, entity_height, 4);
}

module inner_space() {
    overflow = 20;
    translate([top_padding, top_padding, pcb_height-overflow])
    round_cube(pcb_width-top_padding*2, pcb_depth-top_padding*2, top_space_height+overflow, 2);    
}

module top_fan_mouting() {
    difference() {
        union() {
            
            round_cube(top_fan_width+4, top_fan_depth+4, 2, 4, true);
            // cube([top_fan_width+4, top_fan_depth+4, 2], true);
            
            translate([9, -9, 0]) cylinder(6, 8/2, 6/2);
            translate([-9, 9, 0]) cylinder(6, 8/2, 6/2);
        }
        top_fan();
    }
}

module tfcard_tab() {
    top_wall_thickness = wall_thickness + top_padding;
    tab_width = tf_card_width-1;
    translate([pcb_width+wall_thickness-top_wall_thickness, tf_card_pos_y-tab_width/2, 0])
    cube([top_wall_thickness, tab_width, 10]);
}


module step_fitting() {
    difference() {
        translate([-10, -10, 0])
        cube([120, 80, pcb_height + flange_height+step_height]);
        
        edge_step(0.5, -0.4);
    }
}

module a_pcb_screw_mount() {
    rotate([180, 0, 0])
    union() {
        cylinder(5, 8/2, 8/2);
        translate([0, 0, 5])cylinder(3, 8/2, 5/2);
        translate([0, 0, 8]) cylinder(top_space_height-8, 5/2, 5/2);
    }
}

module pcb_screw_mount() {
    for (x = pcbMountingXPos) {
        for (y = pcbMountingYPos){
            translate([x, y, top_space_height+pcb_height]) a_pcb_screw_mount();
        }
    }
}

module box_screws() {
    screw_diameter = 3.0 + 0.5;
    screa_head_diameter = 5.5 + 0.5;
    
    for (x = pcbMountingXPos) {
        for (y = pcbMountingYPos){
            translate([x, y, 0])
            cylinder(100, screw_diameter/2, screw_diameter/2, true);
            
            translate([x, y, 27-14.3])
            cylinder(100, screa_head_diameter/2, screa_head_diameter/2);
        }
    }
}

module air_fin() {
    fin_height = 7;
    translate([cpu_pos_x-28, cpu_pos_y-1-12, top_space_height+pcb_height-fin_height ])
    cube([40, 2, fin_height]);
    
    // offset 3 to avoid conflict with ethernet
    translate([cpu_pos_x-28+3, cpu_pos_y-1+12, top_space_height+pcb_height-fin_height ])
    cube([37, 2, fin_height]);
}

module gpio() {
    hull() {
        translate([60, 9, 0])cylinder(30, 4/2, 4/2);
        translate([70, 9, 0])cylinder(30, 4/2, 4/2);
    }
}

module antenna() {
    translate([18, 2.5, 0])cylinder(30, 3/2, 3/2);
}

module power_cable() {
    translate([76, 9, 0])cylinder(30, 5/2, 5/2);
}

module camera() {
    translate([0, -3, 0]) // offset to avoid conflict with ethernet
    hull() {
        translate([18.56, 33-5.5, 0]) cylinder(30, 1.5/2, 1.5/2);
        translate([18.56, 33+5.5, 0]) cylinder(30, 1.5/2, 1.5/2);
    }
}

module singal_holes() {
    gpio();
    power_cable();
    camera();
    antenna();
}

module top_shell() {
    difference() {
        union() {
            difference() {
                union() {
                    outline();
                }
                
                union() {
                    inner_space();
                    // fan hole
                    translate([cpu_pos_x + top_fan_offset_x, cpu_pos_y-2, pcb_height+top_space_height])
                    cylinder(50, 16.5/2, 16.5/2, true);

                    step_fitting();
                    fix_audio();
                }
            }
            
            // offset2 to make air outlet to cpu
            translate([cpu_pos_x + top_fan_offset_x, cpu_pos_y-2, pcb_height+top_space_height])
            rotate([0, 180, 0])
            top_fan_mouting();
            // tfcard tab
            difference() {
                tfcard_tab();
                fix_pcb_board();
            }
            pcb_screw_mount();
            
            air_fin();
        }
        union() {
            box_screws();
            // mainboard
            mock_mainboard();
            singal_holes();
        }
    }
}

//color([0.3, 0.3, 0.3], 0.7)

top_shell();
// translate([0, 0, -10])
// mock_mainboard();
