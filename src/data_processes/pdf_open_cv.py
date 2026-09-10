import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from typing import List, Tuple, Dict
import os


class PDFContentAnalyzer:
    def __init__(self, image_path: str):
        """
        Initialize the PDF content analyzer with an image of a PDF page.

        Args:
            image_path: Path to the PDF image file
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape

    def preprocess_image(self) -> np.ndarray:
        """
        Preprocess the image for better analysis.

        Returns:
            Preprocessed binary image
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (3, 3), 0)

        # Apply adaptive thresholding for better text detection
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return binary

    def detect_text_regions(
        self, binary_img: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using morphological operations and contour analysis.

        Args:
            binary_img: Binary image

        Returns:
            List of bounding boxes (x, y, w, h) for text regions
        """
        # Create morphological kernel for text detection
        # Horizontal kernel to connect letters in words
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_mask = cv2.morphologyEx(
            binary_img, cv2.MORPH_CLOSE, horizontal_kernel
        )

        # Vertical kernel to connect lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_mask = cv2.morphologyEx(
            horizontal_mask, cv2.MORPH_CLOSE, vertical_kernel
        )

        # Find contours
        contours, _ = cv2.findContours(
            vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on size and aspect ratio typical for text
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            print(x, y, w, h)
            # Text regions are typically wider than they are tall and have reasonable size
            if (
                area > 500
                and aspect_ratio > 2
                and w > 50
                and h > 10
                and h < self.height * 0.8
            ):
                text_regions.append((x, y, w, h))

        return text_regions

    def detect_figure_regions(
        self, binary_img: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect figure regions using edge detection and contour analysis.

        Args:
            binary_img: Binary image

        Returns:
            List of bounding boxes (x, y, w, h) for figure regions
        """
        # Use Canny edge detection
        edges = cv2.Canny(self.gray, 50, 150)

        # Dilate edges to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        figure_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on size and characteristics typical for figures
            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            # Figures tend to be more square-like and larger
            if area > 2000 and 0.2 < aspect_ratio < 5 and w > 80 and h > 80:
                figure_regions.append((x, y, w, h))

        return figure_regions

    def detect_table_regions(
        self, binary_img: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect table regions using line detection.

        Args:
            binary_img: Binary image

        Returns:
            List of bounding boxes (x, y, w, h) for table regions
        """
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(
            binary_img, cv2.MORPH_OPEN, horizontal_kernel
        )

        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel)

        # Combine horizontal and vertical lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

        # Find contours
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on size typical for tables
            area = w * h
            if area > 1000 and w > 100 and h > 100:
                table_regions.append((x, y, w, h))

        return table_regions

    def classify_regions(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Classify different regions in the PDF image.

        Returns:
            Dictionary with classified regions
        """
        binary_img = self.preprocess_image()
        figure_regions = self.detect_figure_regions(binary_img)
        cleaned_regions = {"figures": figure_regions}
        return cleaned_regions

    def rectangles_overlap(self, rect1, rect2, margin=10):
        """
        Check if two rectangles overlap with an optional margin.

        Args:
            rect1, rect2: Tuples of (x, y, w, h)
            margin: Additional margin to consider for overlap

        Returns:
            Boolean indicating if rectangles overlap
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Expand rectangles by margin
        x1 -= margin
        y1 -= margin
        w1 += 2 * margin
        h1 += 2 * margin

        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def visualize_results(
        self,
        regions: Dict[str, List[Tuple[int, int, int, int]]],
        save_path: str = None,
        show_plot: bool = True,
    ):
        """
        Visualize the detected regions on the original image.

        Args:
            regions: Dictionary with classified regions
            save_path: Path to save the visualization (optional)
            show_plot: Whether to display the plot (default: True)
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))

        # Convert BGR to RGB for matplotlib
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb_image)

        # Color mapping for different region types
        colors = {"text": "red", "figures": "blue", "tables": "green"}

        # Draw rectangles for each region type
        for region_type, region_list in regions.items():
            for x, y, w, h in region_list:
                rect = Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=colors[region_type],
                    facecolor="none",
                    label=region_type,
                )
                ax.add_patch(rect)

        # Add legend
        handles = [
            Rectangle(
                (0, 0), 1, 1, edgecolor=color, facecolor="none", label=region_type
            )
            for region_type, color in colors.items()
        ]
        ax.legend(handles=handles, loc="upper right")

        ax.set_title("PDF Content Classification: Text vs Figures vs Tables")
        ax.axis("off")

        plt.tight_layout()

        # Save the plot if save_path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            print(f"Visualization saved to: {save_path}")

        # Show the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_cropped_regions(
        self,
        regions: Dict[str, List[Tuple[int, int, int, int]]],
        output_dir: str = "cropped_regions",
    ):
        """
        Save cropped images of each detected region.

        Args:
            regions: Dictionary with classified regions
            output_dir: Directory to save cropped images
        """

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get base filename for naming cropped images
        base_filename = os.path.splitext(os.path.basename(self.image_path))[0]

        saved_files = []

        for region_type, region_list in regions.items():
            # Create subdirectory for each region type
            region_dir = os.path.join(output_dir, region_type)
            os.makedirs(region_dir, exist_ok=True)

            for i, (x, y, w, h) in enumerate(region_list):
                # Crop the region from the original image
                cropped = self.image[y : y + h, x : x + w]

                # Generate filename
                filename = f"{i+1:03d}.jpg"
                filepath = os.path.join(region_dir, filename)

                # Save the cropped image
                cv2.imwrite(filepath, cropped)
                saved_files.append(filepath)

        print(f"Saved {len(saved_files)} cropped regions to {output_dir}")
        return saved_files

    def get_region_statistics(
        self, regions: Dict[str, List[Tuple[int, int, int, int]]]
    ) -> Dict[str, Dict]:
        """
        Get statistics about detected regions.

        Args:
            regions: Dictionary with classified regions

        Returns:
            Dictionary with statistics for each region type
        """
        stats = {}

        for region_type, region_list in regions.items():
            if region_list:
                areas = [w * h for (x, y, w, h) in region_list]
                stats[region_type] = {
                    "count": len(region_list),
                    "total_area": sum(areas),
                    "avg_area": np.mean(areas),
                    "coverage_percentage": (sum(areas) / (self.width * self.height))
                    * 100,
                }
            else:
                stats[region_type] = {
                    "count": 0,
                    "total_area": 0,
                    "avg_area": 0,
                    "coverage_percentage": 0,
                }

        return stats


def main():
    """
    Main function to run the PDF content analyzer.
    """
    parser = argparse.ArgumentParser(
        description="Analyze PDF image to detect text vs figure regions"
    )
    parser.add_argument("image_path", help="Path to the PDF image file")
    parser.add_argument(
        "--show-stats", action="store_true", help="Show region statistics"
    )
    parser.add_argument(
        "--save-viz", type=str, help="Path to save the visualization image"
    )
    parser.add_argument(
        "--save-crops", type=str, help="Directory to save cropped regions"
    )
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = PDFContentAnalyzer(args.image_path)

        # Classify regions
        print("Analyzing PDF image...")
        regions = analyzer.classify_regions()

        # Display statistics
        if args.show_stats:
            stats = analyzer.get_region_statistics(regions)
            print("\nRegion Statistics:")
            print("-" * 40)
            for region_type, stat in stats.items():
                print(f"{region_type.capitalize()}:")
                print(f"  Count: {stat['count']}")
                print(f"  Coverage: {stat['coverage_percentage']:.1f}%")
                print(f"  Average area: {stat['avg_area']:.0f} pixels")
                print()

        # Visualize results
        print("Displaying results...")
        analyzer.visualize_results(
            regions, save_path=args.save_viz, show_plot=not args.no_show
        )

        # Save cropped regions if requested
        if args.save_crops:
            analyzer.save_cropped_regions(regions, args.save_crops)

        # Return regions for further processing
        return regions

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


if __name__ == "__main__":
    # Example usage without command line arguments
    import os

    # Print current working directory
    current_dir = os.getcwd()
    image_path = "(0 Very Good) Guidance No 38 - Technical guidance for EQS for metals (1) (1).pdf/page_42.jpg"
    # for image_path in os.listdir("output_images"):
    analyzer = PDFContentAnalyzer("output_images/" + image_path)
    regions = analyzer.classify_regions()

    # # Save the visualization
    # analyzer.visualize_results(
    #     regions, save_path=f"analysis_results/{image_path}.png", show_plot=True
    # )

    # Save cropped regions
    analyzer.save_cropped_regions(regions, "analysis_results/cropped_regions")

    # main()
