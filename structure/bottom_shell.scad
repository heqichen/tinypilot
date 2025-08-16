include <mainboard.scad>
include <bottom_fan.scad>
/**

- [x] 1. space for board
- [x] 2. wall for PCB
- [x] 3. heatsink
- [x] 4. fan
- [x] 5. all interfaces
- [x] 6. tf card
- [x] 7. screws for board
- [x] 8. screws for box
- [x] 9. edge with top, step fitting
- [x] 10. airflow route

**/


/**
    
                                                                   ┌┐             
                                                                   ││ step_height  
                                                               ┌───┘│ ────────────
                                                               │    │             
                                                               │    │ flange_height
                                                               │    │             
                                                               │    │             
                        ┌────────────────────────────────────┐ │    │ ────────────
                        │                        PCB         │ │    │ pcb_height   
                0 ─►    └────────────────────────────────────┘ │    │             
                                                            ┌──┘    │ ────────────
                                                            │       │             
                                                            │       │             
                                                            │       │             
                                                            │       │             
                                                            │       │             
                        ────────────────────────────────────┘       │             
                                                                    │             
                                                                    │             
                      ──────────────────────────────────────────────┘             
                                                  
**/

$fn=40;

bottom_fan_cabel_posx = 75;


module fan_cable_protector() {
    $fn=6;
    translate([bottom_fan_cabel_posx, 0, -6])
    intersection() {
        hull() {
            cylinder(3, 2, 4);
            translate([0, 0, 7]) cylinder(3, 4, 2);
        }
        translate([0, -10-wall_thickness + epsilon])cube([20, 20, 40], true);
    }
}

module outline() {
    y_total = bottom_shell_thickness + bottom_space_height + wall_height; 
    
    //color([0.2, 0.2, 0.3], 0.4)
    translate([-wall_thickness, -wall_thickness, -y_total+wall_height])
    round_cube(pcb_width+wall_thickness*2, pcb_depth+wall_thickness*2, y_total, 4);
    
    fan_cable_protector();
}




module pcb_space() {
    round_cube(pcb_width, pcb_depth, 10, 2);
}

module inner_space() {
    translate([0.5, 0.5, -bottom_space_height])
    round_cube(pcb_width-1, pcb_depth-1, bottom_space_height, 2);
}

module pcb_screws() {
    screw_diameter = 3.0 + 0.5;
    difference() {
    intersection() {
        outline();
        union() {
            for (x = pcbMountingXPos) {
                for (y = pcbMountingYPos) {
                    hull() {
                        // bottom enlarged by 3mm
                        if (y > 30) {
                            translate([x, y+20, -bottom_space_height]) cylinder(bottom_space_height, 5/2+3, 5/2);
                        } else {
                            translate([x, y-20, -bottom_space_height]) cylinder(bottom_space_height, 5/2+3, 5/2);
                        }
                        translate([x, y, -bottom_space_height]) cylinder(bottom_space_height, 5/2+3, 5/2);
                    }
                }
            }
        }
    }
    union() {
        for (x = pcbMountingXPos) {
            for (y = pcbMountingYPos) {
                translate([x, y, -bottom_space_height]) cylinder(50, screw_diameter/2, screw_diameter/2, true);
            }
        }
    }
    }   
}


module bottom_fan_mounting_guide() {
    mounting_guide_dimension = 20 + 2*2;
    
    difference() {
        translate([bottom_fan_center_x, bottom_fan_center_y, -bottom_space_height + 1])
        cube([mounting_guide_dimension, mounting_guide_dimension, 2], true);
        
        translate([bottom_fan_center_x, bottom_fan_center_y, -bottom_space_height])
        bottom_fan();
    }
}



module bottom_fan_holes() {
    
    translate([bottom_fan_center_x, bottom_fan_center_y, -bottom_space_height])
    union () {
        // screw holes
        bottom_fan_screw();
        // screw nuts
        translate([0, 0, -1])bottom_fan_nuts();
        // midle hole
        bottom_fan_hole();
    }
    
    // air flow slot
    air_slot_height = 2;
    air_slot_width = 1.5;
    for (y = [-4.5, -1.5, 1.5, 4.5]) {
        translate([0, y, 0])
        translate([0, bottom_fan_center_y, -air_slot_height/2-bottom_space_height-bottom_shell_thickness+air_slot_height])
        cube([200, air_slot_width, air_slot_height], true);
    }
    
    for (x = [-4.5, -1.5, 1.5, 4.5]) {
        translate([x, 0, 0])
        translate([bottom_fan_center_x, 0, -air_slot_height/2-bottom_space_height-bottom_shell_thickness+air_slot_height])
        cube([air_slot_width, 200, air_slot_height], true);
    }
    
    // fan cable
    translate([bottom_fan_cabel_posx-2/2, -0.5, -bottom_space_height]) cube([2, 5, 50]);
    translate([bottom_fan_cabel_posx-2/2, -1.5, -3]) cube([2, 5, 6]);
}

module box_nuts() {
    nut_diameter = 6.0 + 0.5;
    for (x = pcbMountingXPos) {
        for (y = pcbMountingYPos) {
            translate([x, y, -bottom_shell_thickness - bottom_space_height-20 +5]) cylinder(20, nut_diameter/2, nut_diameter/2, $fn=6);
        }
    }
}



module bottom() {
    difference() {
        union() {
            difference() {
                union() {
                    outline();
                }    
                union() {
                    edge_step();
                    pcb_space();
                    inner_space();
                    mock_mainboard();
                }
            }
            bottom_fan_mounting_guide();
            pcb_screws();
        }
        
        bottom_fan_holes();
        fix_audio(false);
        fix_tf_card(false);
        fix_power_button(false);
        fix_pcb_board(false);
        box_nuts();
    }
    
    
    
}


color([0.2, 0.3, 0.4], 0.2) bottom();


// mock_mainboard();
