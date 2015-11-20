//[TIF] Nov. 2015
//       triplet-indexing
//       version 2.0
//This code is used for the following paper:
//Heng Yang*, Renqiao Zhang*, Peter Robinson, 
//"Human and Sheep Landmarks Localisation by Triplet-Interpolated Features", WACV2016
//If you use this code please cite the above publication. 
//Part of the code is taken from dlib.net  Davis E. King (davis@dlib.net)
// The license for dlib.net is : Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICToR_H_
#define DLIB_SHAPE_PREDICToR_H_

#include "shape_predictor_abstract.h"
#include "full_object_detection.h"
#include "../algs.h"
#include "../matrix.h"
#include "../geometry.h"
#include "../pixel.h"
#include "../console_progress_indicator.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        struct split_feature
        {
            unsigned long idx1;
            unsigned long idx2;
            float thresh;

            friend inline void serialize (const split_feature& item, std::ostream& out)
            {
                dlib::serialize(item.idx1, out);
                dlib::serialize(item.idx2, out);
                dlib::serialize(item.thresh, out);
            }
            friend inline void deserialize (split_feature& item, std::istream& in)
            {
                dlib::deserialize(item.idx1, in);
                dlib::deserialize(item.idx2, in);
                dlib::deserialize(item.thresh, in);
            }
        };


        // a tree is just a std::vector<impl::split_feature>.  We use this function to navigate the
        // tree nodes
        inline unsigned long left_child (unsigned long idx) { return 2*idx + 1; }
        /*!
            ensures
                - returns the index of the left child of the binary tree node idx
        !*/
        inline unsigned long right_child (unsigned long idx) { return 2*idx + 2; }
        /*!
            ensures
                - returns the index of the left child of the binary tree node idx
        !*/

        struct regression_tree
        {
            //[TIF] arrays of splits and leaf_values
            std::vector<split_feature> splits;
            std::vector<matrix<float,0,1> > leaf_values;


            //[TIF] go through a tree, for the given feature_pixel_values
            inline const matrix<float,0,1>& operator()(    
                const std::vector<float>& feature_pixel_values
            ) const
            /*!
                requires
                    - All the index values in splits are less than feature_pixel_values.size()
                    - leaf_values.size() is a power of 2.
                      (i.e. we require a tree with all the levels fully filled out.
                    - leaf_values.size() == splits.size()+1
                      (i.e. there needs to be the right number of leaves given the number of splits in the tree)
                ensures
                    - runs through the tree and returns the vector at the leaf we end up in.
            !*/
            {
                unsigned long i = 0;
                while (i < splits.size())
                {
                    if (feature_pixel_values[splits[i].idx1] - feature_pixel_values[splits[i].idx2] > splits[i].thresh)
                        i = left_child(i);
                    else
                        i = right_child(i);
                }
                return leaf_values[i - splits.size()];
            }

            friend void serialize (const regression_tree& item, std::ostream& out)
            {
                dlib::serialize(item.splits, out);
                dlib::serialize(item.leaf_values, out);
            }
            friend void deserialize (regression_tree& item, std::istream& in)
            {
                dlib::deserialize(item.splits, in);
                dlib::deserialize(item.leaf_values, in);
            }
        };




    // ------------------------------------------------------------------------------------

        inline vector<float,2> location (
            const matrix<float,0,1>& shape,
            unsigned long idx
        )
        /*!
            requires
                - idx < shape.size()/2
                - shape.size()%2 == 0
            ensures
                - returns the idx-th point from the shape vector.
        !*/
        {
            return vector<float,2>(shape(idx*2), shape(idx*2+1));
        }


    // ------------------------------------------------------------------------------------


        class index_feature
        {
        public:

            inline std::vector<unsigned long> anchor(unsigned long id)
            {
                std::vector<unsigned long> id_xyz;
                id_xyz.push_back(anchor_idx[id]);
                id_xyz.push_back(anchor_idy[id]);
                id_xyz.push_back(anchor_idz[id]);
                return id_xyz;
            }

            inline std::vector<double> ratio (unsigned long id)
            {
                std::vector<double> ratio_out;
                ratio_out.push_back(ratio_a[id]);
                ratio_out.push_back(ratio_b[id]);
                return ratio_out;
            }

            inline unsigned long get_num_of_anchors()
            const
            {
                return anchor_idx.size();
            }

            void assign (
                const unsigned long i, 
                const unsigned long id0, const unsigned long id1, const unsigned long id2, 
                const double ratio1, const double ratio2 )
            {
                anchor_idx[i] = id0;
                anchor_idy[i] = id1;
                anchor_idz[i] = id2;
                ratio_a[i] = ratio1;
                ratio_b[i] = ratio2;
            }

            void set_size (unsigned long newsize)
            {
                anchor_idx.resize (newsize);
                anchor_idy.resize (newsize);
                anchor_idz.resize (newsize);
                ratio_a.resize (newsize);
                ratio_b.resize (newsize);
            }

            inline dlib::vector<float, 2> p_location (const matrix<float, 0,1>& shape, const unsigned long i)
            const
            {
                unsigned long idx = anchor_idx[i];
                unsigned long idy = anchor_idy[i];
                unsigned long idz = anchor_idz[i];
                double a = ratio_a[i];      double b = ratio_b[i];
                dlib::vector<float,2> p_coords;
        
                p_coords.x() = a*(location(shape,idy)[0]-location(shape,idx)[0]) + b*(location(shape,idz)[0]-location(shape,idx)[0]) + location(shape, idx)[0];
                p_coords.y() = a*(location(shape,idy)[1]-location(shape,idx)[1]) + b*(location(shape,idz)[1]-location(shape,idx)[1]) + location(shape, idx)[1];

                return p_coords;
            }

            friend inline void serialize (const index_feature& item, std::ostream& out)
            {
                dlib::serialize(item.anchor_idx, out);
                dlib::serialize(item.anchor_idy, out);
                dlib::serialize(item.anchor_idz, out);
                dlib::serialize(item.ratio_a, out);
                dlib::serialize(item.ratio_b, out);
            }

            friend inline void deserialize (index_feature& item, std::istream& in)
            {
                dlib::deserialize(item.anchor_idx, in);
                dlib::deserialize(item.anchor_idy, in);
                dlib::deserialize(item.anchor_idz, in);
                dlib::deserialize(item.ratio_a, in);
                dlib::deserialize(item.ratio_b, in);
            }


        private:
            std::vector<unsigned long> anchor_idx;
            std::vector<unsigned long> anchor_idy;
            std::vector<unsigned long> anchor_idz;
            std::vector<double> ratio_a;
            std::vector<double> ratio_b;

            inline std::vector<float> location (
                const matrix<float, 0,1>& shape,
                unsigned long idx) const
            {
                std::vector<float> loc;
                loc.push_back(  shape(idx*2)  );
                loc.push_back( shape(idx*2+1) );
                return loc;
            }

        };
    // ------------------------------------------------------------------------------------

        //[ANDY] inline unsigned long nearest_shape_point ();

    // ------------------------------------------------------------------------------------

        //[ANDY] inline void create_shape_relative_encoding ();

    // ------------------------------------------------------------------------------------

        inline point_transform_affine find_tform_between_shapes (
            const matrix<float,0,1>& from_shape,
            const matrix<float,0,1>& to_shape
        )
        {
            DLIB_ASSERT(from_shape.size() == to_shape.size() && (from_shape.size()%2) == 0 && from_shape.size() > 0,"");
            std::vector<vector<float,2> > from_points, to_points;
            const unsigned long num = from_shape.size()/2;
            from_points.reserve(num);
            to_points.reserve(num);
            if (num == 1)
            {
                // Just use an identity transform if there is only one landmark.
                return point_transform_affine();
            }

            for (unsigned long i = 0; i < num; ++i)
            {
                from_points.push_back(location(from_shape,i));
                to_points.push_back(location(to_shape,i));
            }
            return find_similarity_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

        inline point_transform_affine normalizing_tform (
            const rectangle& rect
        )
        /*!
            ensures
                - returns a transform that maps rect.tl_corner() to (0,0) and rect.br_corner()
                  to (1,1).
        !*/
        {
            std::vector<vector<float,2> > from_points, to_points;
            from_points.push_back(rect.tl_corner()); to_points.push_back(point(0,0));
            from_points.push_back(rect.tr_corner()); to_points.push_back(point(1,0));
            from_points.push_back(rect.br_corner()); to_points.push_back(point(1,1));
            return find_affine_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

        inline point_transform_affine unnormalizing_tform (
            const rectangle& rect
        )
        /*!
            ensures
                - returns a transform that maps (0,0) to rect.tl_corner() and (1,1) to
                  rect.br_corner().
        !*/
        {
            std::vector<vector<float,2> > from_points, to_points;
            to_points.push_back(rect.tl_corner()); from_points.push_back(point(0,0));
            to_points.push_back(rect.tr_corner()); from_points.push_back(point(1,0));
            to_points.push_back(rect.br_corner()); from_points.push_back(point(1,1));
            return find_affine_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

        template <typename image_type>
        void extract_feature_pixel_values (
            //[ANDY] input>>
            const image_type& img_,
            const rectangle& rect,
            const matrix<float,0,1>& current_shape,

            const index_feature& index,

            //[ANDY] output<<
            std::vector<float>& feature_pixel_values
        )
        //[ANDY] extract the values of one single image/box
        //       columns of ref_anchor_idx and ratio correspond to feature_pixel_values by the parallel column/array index

        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - reference_pixel_anchor_idx.size() == reference_pixel_deltas.size()
                - current_shape.size() == reference_shape.size()
                - reference_shape.size()%2 == 0
                - max(mat(reference_pixel_anchor_idx)) < reference_shape.size()/2
            ensures
                - #feature_pixel_values.size() == reference_pixel_deltas.size()
                - for all valid i:
                    - #feature_pixel_values[i] == the value of the pixel in img_ that
                      corresponds to the pixel identified by reference_pixel_anchor_idx[i]
                      and reference_pixel_deltas[i] when the pixel is located relative to
                      current_shape rather than reference_shape.
        !*/
        {
            //[ANDY] tform_to_img: shape -> rect (img)
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);

            const rectangle area = get_rect(img_);
            const_image_view<image_type> img(img_);

            //[ANDY] #_of_feature_pixels = #_of_refAnchorIDX_triplets = #_of_ratio_pairs 
            feature_pixel_values.resize( index.get_num_of_anchors() );

            for (unsigned long i = 0; i < feature_pixel_values.size(); ++i)
            {
                // Compute the point in the current shape corresponding to the i-th pixel and
                // then map it from the normalized shape space into pixel space.
                point p = tform_to_img( index.p_location(current_shape, i) );
                if (area.contains(p))
                    feature_pixel_values[i] = get_pixel_intensity( img [p.y()][p.x()] );
                else
                    feature_pixel_values[i] = 0;
            }
        }

    } // end namespace impl

// ----------------------------------------------------------------------------------------

    class shape_predictor
    {
    public:


        shape_predictor (
        ) 
        {}

        shape_predictor (

            const matrix<float,0,1>& initial_shape_,
            const std::vector<std::vector<impl::regression_tree> >& forests_,

            const std::vector<impl::index_feature> index_

        ) : initial_shape(initial_shape_), forests(forests_), index(index_)

        //[ANDY] this constructor generates a shape_predictor, 
        //       consisting forests/initial_shape/anchor_idx/ratio

        /*!
            requires
                - initial_shape.size()%2 == 0
                - forests.size() == pixel_coordinates.size() == the number of cascades
                - for all valid i:
                    - all the index values in forests[i] are less than pixel_coordinates[i].size()
                - for all valid i and j: 
                    - forests[i][j].leaf_values.size() is a power of 2.
                      (i.e. we require a tree with all the levels fully filled out.
                    - forests[i][j].leaf_values.size() == forests[i][j].splits.size()+1
                      (i.e. there need to be the right number of leaves given the number of splits in the tree)
        !*/
        {}

        unsigned long num_parts (
        ) const
        {
            std::cout << "test" << std::endl;
            return initial_shape.size()/2;
        }

        template <typename image_type>
        full_object_detection operator()(
            const image_type& img,
            const rectangle& rect
        ) const
        {
            using namespace impl;
            matrix<float,0,1> current_shape = initial_shape;
            std::vector<float> feature_pixel_values;

            //[ANDY] iter->cascade, i->individual trees
            for (unsigned long iter = 0; iter < forests.size(); ++iter)
            {
                extract_feature_pixel_values(img, rect, current_shape, index[iter], feature_pixel_values);
                // evaluate all the trees at this level of the cascade.
                for (unsigned long i = 0; i < forests[iter].size(); ++i)
                    current_shape += forests[iter][i](feature_pixel_values);
            }

            // convert the current_shape into a full_object_detection
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);

            std::vector<point> parts(current_shape.size()/2);

            for (unsigned long i = 0; i < parts.size(); ++i)
                parts[i] = tform_to_img(location(current_shape, i));
            return full_object_detection(rect, parts);
        }

        friend void serialize (const shape_predictor& item, std::ostream& out)
        {
            int version = 1;
            dlib::serialize(version, out);
            dlib::serialize(item.initial_shape, out);
            dlib::serialize(item.forests, out);
            dlib::serialize(item.index, out);
        }
        friend void deserialize (shape_predictor& item, std::istream& in)
        {
            int version = 0;
            dlib::deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::shape_predictor.");
            dlib::deserialize(item.initial_shape, in);
            dlib::deserialize(item.forests, in);
            dlib::deserialize(item.index, in);
        }

    private:
        matrix<float,0,1> initial_shape;
        std::vector< std::vector<impl::regression_tree> > forests;
        std::vector< impl::index_feature > index;
    };

// ----------------------------------------------------------------------------------------

    class shape_predictor_trainer
    {
        /*!
            This thing really only works with unsigned char or rgb_pixel images (since we assume the threshold 
            should be in the range [-128,128]).
        !*/
    public:

        shape_predictor_trainer (
        )
        {
            _cascade_depth = 10;
            _tree_depth = 4;
            _num_trees_per_cascade_level = 500;
            _nu = 0.1;
            _oversampling_amount = 20;
            _feature_pool_size = 400;
            _lambda = 0.1;
            _num_test_splits = 20;
            _feature_pool_region_padding = 0;
            _verbose = false;
        }

        unsigned long get_cascade_depth (
        ) const { return _cascade_depth; }

        void set_cascade_depth (
            unsigned long depth
        )
        {
            DLIB_CASSERT(depth > 0, 
                "\t void shape_predictor_trainer::set_cascade_depth()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t depth:  " << depth
            );

            _cascade_depth = depth;
        }

        unsigned long get_tree_depth (
        ) const { return _tree_depth; }

        void set_tree_depth (
            unsigned long depth
        )
        {
            DLIB_CASSERT(depth > 0, 
                "\t void shape_predictor_trainer::set_tree_depth()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t depth:  " << depth
            );

            _tree_depth = depth;
        }

        unsigned long get_num_trees_per_cascade_level (
        ) const { return _num_trees_per_cascade_level; }

        void set_num_trees_per_cascade_level (
            unsigned long num
        )
        {
            DLIB_CASSERT( num > 0,
                "\t void shape_predictor_trainer::set_num_trees_per_cascade_level()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t num:  " << num
            );
            _num_trees_per_cascade_level = num;
        }

        double get_nu (
        ) const { return _nu; } 
        void set_nu (
            double nu
        )
        {
            DLIB_CASSERT(0 < nu && nu <= 1,
                "\t void shape_predictor_trainer::set_nu()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t nu:  " << nu 
            );

            _nu = nu;
        }

        std::string get_random_seed (
        ) const { return rnd.get_seed(); }
        void set_random_seed (
            const std::string& seed
        ) { rnd.set_seed(seed); }

        unsigned long get_oversampling_amount (
        ) const { return _oversampling_amount; }
        void set_oversampling_amount (
            unsigned long amount
        )
        {
            DLIB_CASSERT(amount > 0, 
                "\t void shape_predictor_trainer::set_oversampling_amount()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t amount: " << amount 
            );

            _oversampling_amount = amount;
        }

        unsigned long get_feature_pool_size (
        ) const { return _feature_pool_size; }
        void set_feature_pool_size (
            unsigned long size
        ) 
        {
            DLIB_CASSERT(size > 1, 
                "\t void shape_predictor_trainer::set_feature_pool_size()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t size: " << size 
            );

            _feature_pool_size = size;
        }

        double get_lambda (
        ) const { return _lambda; }
        void set_lambda (
            double lambda
        )
        {
            DLIB_CASSERT(lambda > 0,
                "\t void shape_predictor_trainer::set_lambda()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t lambda: " << lambda 
            );

            _lambda = lambda;
        }

        unsigned long get_num_test_splits (
        ) const { return _num_test_splits; }
        void set_num_test_splits (
            unsigned long num
        )
        {
            DLIB_CASSERT(num > 0, 
                "\t void shape_predictor_trainer::set_num_test_splits()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t num: " << num 
            );

            _num_test_splits = num;
        }


        double get_feature_pool_region_padding (
        ) const { return _feature_pool_region_padding; }
        void set_feature_pool_region_padding (
            double padding 
        )
        {
            _feature_pool_region_padding = padding;
        }

        void be_verbose (
        )
        {
            _verbose = true;
        }

        void be_quiet (
        )
        {
            _verbose = false;
        }

        template <typename image_array>
        shape_predictor train (

            //[ANDY] input >> images + objects = labelled data
            const image_array& images,
            const std::vector<std::vector<full_object_detection> >& objects

        ) const

        {

            using namespace impl;
            DLIB_CASSERT(
                images.size() == objects.size() && images.size() > 0,
                "\t shape_predictor shape_predictor_trainer::train()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t images.size():  " << images.size() 
                << "\n\t objects.size(): " << objects.size() 
            );

            // make sure the objects agree on the number of parts and that there is at
            // least one full_object_detection. 
            unsigned long num_parts = 0;
            for (unsigned long i = 0; i < objects.size(); ++i)
            {
                for (unsigned long j = 0; j < objects[i].size(); ++j)
                {
                    if (num_parts == 0)
                    {
                        num_parts = objects[i][j].num_parts();
                        DLIB_CASSERT(objects[i][j].num_parts() != 0,
                            "\t shape_predictor shape_predictor_trainer::train()"
                            << "\n\t You can't give objects that don't have any parts to the trainer."
                        );
                    }
                    else
                    {
                        DLIB_CASSERT(objects[i][j].num_parts() == num_parts,
                            "\t shape_predictor shape_predictor_trainer::train()"
                            << "\n\t All the objects must agree on the number of parts. "
                            << "\n\t objects["<<i<<"]["<<j<<"].num_parts(): " << objects[i][j].num_parts()
                            << "\n\t num_parts:  " << num_parts 
                        );
                    }
                }
            }
            DLIB_CASSERT(num_parts != 0,
                "\t shape_predictor shape_predictor_trainer::train()"
                << "\n\t You must give at least one full_object_detection if you want to train a shape model and it must have parts."
            );



            rnd.set_seed(get_random_seed());

            std::vector<training_sample> samples;

            //[ANDY] initial shape, generated by averaging training samples
            const matrix<float,0,1> initial_shape = populate_training_sample_shapes(objects, samples);


            //[ANDY] adjustment
            //       instead of pixel_coordinates, we generate anchor_idx(3-long vector) and ratio to denote an indexed point 
            std::vector< impl::index_feature > index;


            //[ANDY] create indexing for all cascade
            randomly_sample_pixel_coordinates( index, initial_shape);


            unsigned long trees_fit_so_far = 0;
            console_progress_indicator pbar(get_cascade_depth()*get_num_trees_per_cascade_level());
            if (_verbose)
                std::cout << "Fitting trees..." << std::endl;


            std::vector<std::vector<impl::regression_tree> > forests(get_cascade_depth());
            // Now start doing the actual training by filling in the forests
            for (unsigned long cascade = 0; cascade < get_cascade_depth(); ++cascade)
            {
                // Each cascade uses a different set of pixels for its features. 

                //[ANDY] First compute all the feature_pixel_values for each training sample at this level of the cascade.
                //       no encoding needed
                //       run through each sample
                for (unsigned long i = 0; i < samples.size(); ++i)
                {
                    extract_feature_pixel_values(
                        images[samples[i].image_idx], samples[i].rect, samples[i].current_shape, 
                        index[cascade], samples[i].feature_pixel_values 
                    );
                }

                // Now start building the trees at this cascade level.
                for (unsigned long i = 0; i < get_num_trees_per_cascade_level(); ++i)
                {
                    forests[cascade].push_back( 
                        make_regression_tree( samples, initial_shape, index[cascade] )
                    );

                    if (_verbose)
                    {
                        ++trees_fit_so_far;
                        pbar.print_status(trees_fit_so_far);
                    }
                }
            }

            if (_verbose)
                std::cout << "Training complete                          " << std::endl;

            return shape_predictor( initial_shape, forests, index );
        }


/*      [ANDY]
            shape_predictor (

            const matrix<float,0,1>& initial_shape_,
            const std::vector<std::vector<impl::regression_tree> >& forests_,

            const std::vector<matrix<unsigned long, 0,3> >& anchor_idx,
            const std::vector<matrix<double, 0,2>& ratio 
*/

    private:

        static matrix<float,0,1> object_to_shape (
            const full_object_detection& obj
        )
        {
            matrix<float,0,1> shape(obj.num_parts()*2);
            const point_transform_affine tform_from_img = impl::normalizing_tform(obj.get_rect());
            for (unsigned long i = 0; i < obj.num_parts(); ++i)
            {
                vector<float,2> p = tform_from_img(obj.part(i));
                shape(2*i)   = p.x();
                shape(2*i+1) = p.y();
            }
            return shape;
        }

        struct training_sample 
        {
            /*!

            CONVENTION
                - feature_pixel_values.size() == get_feature_pool_size()
                - feature_pixel_values[j] == the value of the j-th feature pool
                  pixel when you look it up relative to the shape in current_shape.

                - target_shape == The truth shape.  Stays constant during the whole
                  training process.
                - rect == the position of the object in the image_idx-th image.  All shape
                  coordinates are coded relative to this rectangle.
            !*/

            unsigned long image_idx;
            rectangle rect;
            matrix<float,0,1> target_shape; 

            matrix<float,0,1> current_shape;  
            std::vector<float> feature_pixel_values;

            void swap(training_sample& item)
            {
                std::swap(image_idx, item.image_idx);
                std::swap(rect, item.rect);
                target_shape.swap(item.target_shape);
                current_shape.swap(item.current_shape);
                feature_pixel_values.swap(item.feature_pixel_values);
            }
        };

        impl::regression_tree make_regression_tree (
            std::vector<training_sample>& samples,
            const matrix<float, 0,1>& shape,
            const impl::index_feature& index

        ) const

        {
            using namespace impl;
            std::deque<std::pair<unsigned long, unsigned long> > parts;
            parts.push_back(std::make_pair(0, (unsigned long)samples.size()));

            impl::regression_tree tree;

            // walk the tree in breadth first order
            const unsigned long num_split_nodes = static_cast<unsigned long>(std::pow(2.0, (double)get_tree_depth())-1);
            std::vector<matrix<float,0,1> > sums(num_split_nodes*2+1);
            for (unsigned long i = 0; i < samples.size(); ++i)
                sums[0] += samples[i].target_shape - samples[i].current_shape;

            for (unsigned long i = 0; i < num_split_nodes; ++i) 
            {
                std::pair<unsigned long,unsigned long> range = parts.front();
                parts.pop_front();

                //[ANDY] using new generate_split function
                const impl::split_feature split = generate_split
                (
                    samples, range.first,range.second, 
                    shape, index,  
                    sums[i], sums[left_child(i)], sums[right_child(i)]
                );
                tree.splits.push_back(split);


                const unsigned long mid = partition_samples(split, samples, range.first, range.second); 

                parts.push_back(std::make_pair(range.first, mid));
                parts.push_back(std::make_pair(mid, range.second));
            }

            // Now all the parts contain the ranges for the leaves so we can use them to
            // compute the average leaf values.
            tree.leaf_values.resize(parts.size());
            for (unsigned long i = 0; i < parts.size(); ++i)
            {
                if (parts[i].second != parts[i].first)
                    tree.leaf_values[i] = sums[num_split_nodes+i]*get_nu()/(parts[i].second - parts[i].first);
                else
                    tree.leaf_values[i] = zeros_matrix(samples[0].target_shape);

                // now adjust the current shape based on these predictions
                for (unsigned long j = parts[i].first; j < parts[i].second; ++j)
                    samples[j].current_shape += tree.leaf_values[i];
            }

            return tree;
        }


        impl::split_feature randomly_generate_split_feature (

            //[ANDY] outpu >>
            const impl::index_feature& index,
        
            //[ANDY] input >>
            const matrix<float, 0,1>& shape
        ) const

        //[ANDY] generate split feature for one cascade, pixel_coordinates is replaced by ratio and anchor
        {
            impl::split_feature feat;

            const double lambda = get_lambda(); 
            double accept_prob;
            do 
            {
                feat.idx1   = rnd.get_random_32bit_number()%get_feature_pool_size();
                feat.idx2   = rnd.get_random_32bit_number()%get_feature_pool_size();

                //[ANDY] dist = ||u-v||
                const double dist = length(
                    index.p_location(shape, feat.idx1) - index.p_location(shape, feat.idx2)
                );

                accept_prob = std::exp(-dist/lambda);
            }
            while( feat.idx1 == feat.idx2 || !(accept_prob > rnd.get_random_double()));

            feat.thresh = (rnd.get_random_double()*256 - 128)/2.0;

            return feat;
        }


        impl::split_feature generate_split (
            const std::vector<training_sample>& samples,
            unsigned long begin,
            unsigned long end,

            const matrix<float, 0,1>& shape,
            const impl::index_feature& index,
            
            const matrix<float,0,1>& sum,
            matrix<float,0,1>& left_sum,
            matrix<float,0,1>& right_sum 
        ) const
        {
            // generate a bunch of random splits and test them and return the best one.
            const unsigned long num_test_splits = get_num_test_splits();  

            // sample the random features we test in this function
            std::vector<impl::split_feature> feats;
            feats.reserve(num_test_splits);
            for ( unsigned long i = 0; i < num_test_splits; ++i )
                feats.push_back( randomly_generate_split_feature( index, shape ) );

            std::vector<matrix<float,0,1> > left_sums(num_test_splits);
            std::vector<unsigned long> left_cnt(num_test_splits);

            // now compute the sums of vectors that go left for each feature
            matrix<float,0,1> temp;
            for (unsigned long j = begin; j < end; ++j)
            {
                temp = samples[j].target_shape - samples[j].current_shape;
                for (unsigned long i = 0; i < num_test_splits; ++i)
                {
                    if (samples[j].feature_pixel_values[feats[i].idx1] - samples[j].feature_pixel_values[feats[i].idx2] > feats[i].thresh)
                    {
                        left_sums[i] += temp;
                        ++left_cnt[i];
                    }
                }
            }

            // now figure out which feature is the best
            double best_score = -1;
            unsigned long best_feat = 0;
            for (unsigned long i = 0; i < num_test_splits; ++i)
            {
                // check how well the feature splits the space.
                double score = 0;
                unsigned long right_cnt = end-begin-left_cnt[i];
                if (left_cnt[i] != 0 && right_cnt != 0)
                {
                    temp = sum - left_sums[i];
                    score = dot(left_sums[i],left_sums[i])/left_cnt[i] + dot(temp,temp)/right_cnt;
                    if (score > best_score)
                    {
                        best_score = score;
                        best_feat = i;
                    }
                }
            }

            left_sums[best_feat].swap(left_sum);
            if (left_sum.size() != 0)
            {
                right_sum = sum - left_sum;
            }
            else
            {
                right_sum = sum;
                left_sum = zeros_matrix(sum);
            }
            return feats[best_feat];
        }

        unsigned long partition_samples (
            const impl::split_feature& split,
            std::vector<training_sample>& samples,
            unsigned long begin,
            unsigned long end
        ) const
        {
            // splits samples based on split (sorta like in quick sort) and returns the mid
            // point.  make sure you return the mid in a way compatible with how we walk
            // through the tree.

            unsigned long i = begin;
            for (unsigned long j = begin; j < end; ++j)
            {
                if (samples[j].feature_pixel_values[split.idx1] - samples[j].feature_pixel_values[split.idx2] > split.thresh)
                {
                    samples[i].swap(samples[j]);
                    ++i;
                }
            }
            return i;
        }



        matrix<float,0,1> populate_training_sample_shapes(
            const std::vector<std::vector<full_object_detection> >& objects,
            std::vector<training_sample>& samples
        ) const
        {
            samples.clear();
            matrix<float,0,1> mean_shape;
            long count = 0;
            // first fill out the target shapes
            for (unsigned long i = 0; i < objects.size(); ++i)
            {
                for (unsigned long j = 0; j < objects[i].size(); ++j)
                {
                    training_sample sample;
                    sample.image_idx = i;
                    sample.rect = objects[i][j].get_rect();
                    sample.target_shape = object_to_shape(objects[i][j]);
                    for (unsigned long itr = 0; itr < get_oversampling_amount(); ++itr)
                        samples.push_back(sample);
                    mean_shape += sample.target_shape;
                    ++count;
                }
            }

            mean_shape /= count;

            // now go pick random initial shapes
            for (unsigned long i = 0; i < samples.size(); ++i)
            {
                if ((i%get_oversampling_amount()) == 0)
                {
                    // The mean shape is what we really use as an initial shape so always
                    // include it in the training set as an example starting shape.
                    samples[i].current_shape = mean_shape;
                }
                else
                {
                    // Pick a random convex combination of two of the target shapes and use
                    // that as the initial shape for this sample.
                    const unsigned long rand_idx = rnd.get_random_32bit_number()%samples.size();
                    const unsigned long rand_idx2 = rnd.get_random_32bit_number()%samples.size();
                    const double alpha = rnd.get_random_double();
                    samples[i].current_shape = alpha*samples[rand_idx].target_shape + (1-alpha)*samples[rand_idx2].target_shape;
                }
            }


            return mean_shape;
        }


        void sample_pixel_coordinates (
            //-------------------------------------------
            impl::index_feature& index,
            //-------------------------------------------
            const matrix<float, 0,1>& shape
        ) const

        {
            index.set_size( get_feature_pool_size() );

            for (unsigned long i = 0; i < get_feature_pool_size(); ++i) 
            {
                double alpha_new = rnd.get_random_double()*(0.5);           
                double beta_new  = rnd.get_random_double()*(0.5);

                unsigned long id0;  unsigned long id1;  unsigned long id2;

                do
                {
                    id0 = rnd.get_random_32bit_number() % ( (shape.size())/2 );
                    id1 = rnd.get_random_32bit_number() % ( (shape.size())/2 );
                    id2 = rnd.get_random_32bit_number() % ( (shape.size())/2 );
                }
                while ( id0==id1 || id1==id2 || id2==id0 );

                index.assign( i, id0, id1, id2, alpha_new, beta_new );

            }
        }

        





        void randomly_sample_pixel_coordinates (
            std::vector< impl::index_feature >& index,
            const matrix<float,0,1>& initial_shape
        ) const
        {
            index.resize(get_cascade_depth() );

            //[ANDY] index a new set of point for each cascade
            for (unsigned long i = 0; i < get_cascade_depth(); ++i)
            {                
                sample_pixel_coordinates( index[i], initial_shape );
            }
        }


        mutable dlib::rand rnd;

        unsigned long _cascade_depth;
        unsigned long _tree_depth;
        unsigned long _num_trees_per_cascade_level;
        double _nu;
        unsigned long _oversampling_amount;
        unsigned long _feature_pool_size;
        double _lambda;
        unsigned long _num_test_splits;
        double _feature_pool_region_padding;
        bool _verbose;
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename image_array
        >
    double test_shape_predictor (
        const shape_predictor& sp,
        const image_array& images,
        const std::vector<std::vector<full_object_detection> >& objects,
        const std::vector<std::vector<double> >& scales
    )
    {
        // make sure requires clause is not broken
#ifdef ENABLE_ASSERTS
        DLIB_CASSERT( images.size() == objects.size() ,
            "\t double test_shape_predictor()"
            << "\n\t Invalid inputs were given to this function. "
            << "\n\t images.size():  " << images.size() 
            << "\n\t objects.size(): " << objects.size() 
        );
        for (unsigned long i = 0; i < objects.size(); ++i)
        {
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                DLIB_CASSERT(objects[i][j].num_parts() == sp.num_parts(), 
                    "\t double test_shape_predictor()"
                    << "\n\t Invalid inputs were given to this function. "
                    << "\n\t objects["<<i<<"]["<<j<<"].num_parts(): " << objects[i][j].num_parts()
                    << "\n\t sp.num_parts(): " << sp.num_parts()
                );
            }
            if (scales.size() != 0)
            {
                DLIB_CASSERT(objects[i].size() == scales[i].size(), 
                    "\t double test_shape_predictor()"
                    << "\n\t Invalid inputs were given to this function. "
                    << "\n\t objects["<<i<<"].size(): " << objects[i].size()
                    << "\n\t scales["<<i<<"].size(): " << scales[i].size()
                );

            }
        }
#endif

        running_stats<double> rs;
        for (unsigned long i = 0; i < objects.size(); ++i)
        {
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                // Just use a scale of 1 (i.e. no scale at all) if the caller didn't supply
                // any scales.
                const double scale = scales.size()==0 ? 1 : scales[i][j]; 

                full_object_detection det = sp(images[i], objects[i][j].get_rect());

                for (unsigned long k = 0; k < det.num_parts(); ++k)
                {
                    double score = length(det.part(k) - objects[i][j].part(k))/scale;
                    rs.add(score);
                }
            }
        }
        return rs.mean();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_array
        >
    double test_shape_predictor (
        const shape_predictor& sp,
        const image_array& images,
        const std::vector<std::vector<full_object_detection> >& objects
    )
    {
        std::vector<std::vector<double> > no_scales;
        return test_shape_predictor(sp, images, objects, no_scales);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHAPE_PREDICToR_H_

